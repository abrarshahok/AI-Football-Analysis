from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.kMeans = None
        self.team_colors = {}
        self.player_team = {}

    def get_cluster_model(self, image):
        # reshape image into 2d
        image_2d = image.reshape(-1, 3)

        # perform clustering
        kMeans =  KMeans(n_clusters=2, init='k-means++', n_init=1)
        kMeans.fit(image_2d)

        return kMeans

    def get_cluster_centers(self, kMeans, clustered_image):
        # get corner clusters to differentiate b/w player and non-player clusters
        top_left = clustered_image[0, 0]
        top_right = clustered_image[0, -1]
        bottom_left = clustered_image[-1, 0]
        bottom_right = clustered_image[-1, -1]

        # get the max value from corner and get player cluster
        corner_clusters = [top_left, top_right, bottom_left, bottom_right]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # get cluster centers using player cluster
        cluster_centers = kMeans.cluster_centers_[player_cluster]

        return cluster_centers


    def get_player_color(self, frame, bbox):
        # get bbox co-ordinates
        x1, y1, x2, y2 = bbox

        # get image inside bounding box
        image = frame[int(y1): int(y2), int(x1): int(x2)]

        # get top half of image (T-Shirt only)
        n = image.shape[0] // 2
        top_half_image = image[:n, :]

        # get cluster model
        kMeans = self.get_cluster_model(top_half_image)

        # get the cluster labels
        labels = kMeans.labels_

        # get height and width
        height, width = top_half_image.shape[0], top_half_image.shape[1]

        # reshape to original image
        clustered_image = labels.reshape(height, width)

        # get cluser centers as rgb
        player_color_rgb = self.get_cluster_centers(kMeans, clustered_image)

        return player_color_rgb


    def assign_team_color(self, frame, player_detections):
        # store player colors in list
        player_colors = []

        # get color for each player
        for _, player in player_detections.items():
            bbox = player['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # divide players into team colors
        kMeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kMeans.fit(player_colors)

        # storing kMeans for future use
        self.kMeans = kMeans

        # one team will be assigned green color and other team will be assigned white color
        self.team_colors[1] = kMeans.cluster_centers_[0]
        self.team_colors[2] = kMeans.cluster_centers_[1]
    
    def get_player_team(self, frame, player_bbox, player_id):
        # avoid redundant assignments
        if player_id in self.player_team:
            return self.player_team[player_id]  

        # get player color
        player_color = self.get_player_color(frame, player_bbox)

        # predict which team this color belongs to
        predicted_team_id = self.kMeans.predict(player_color.reshape(1, -1))[0]

        # team id is 1 if predicted id is 0 else 2
        team_id = 1 if predicted_team_id == 0 else 2

        # save team_id in player team
        self.player_team[player_id] = team_id

        return team_id
