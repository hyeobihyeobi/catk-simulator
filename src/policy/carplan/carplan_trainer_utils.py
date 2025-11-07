"""
stationary_at_crosswalk
starting_unprotected_noncross_turn
behind_pedestrian_on_driveable
high_lateral_acceleration
low_magnitude_speed
stationary_at_traffic_light_with_lead
near_multiple_vehicles
starting_protected_noncross_turn
on_stopline_crosswalk
accelerating_at_traffic_light_with_lead
medium_magnitude_speed
on_carpark
near_multiple_pedestrians
near_long_vehicle
on_pickup_dropoff
traversing_pickup_dropoff
traversing_crosswalk
high_magnitude_speed
following_lane_with_slow_lead
on_traffic_light_intersection
traversing_intersection
near_trafficcone_on_driveable
stationary
starting_protected_cross_turn
stopping_at_traffic_light_without_lead
changing_lane_to_left
near_pedestrian_on_crosswalk
on_stopline_stop_sign
waiting_for_pedestrian_to_cross
stopping_with_lead
changing_lane_to_right
traversing_narrow_lane
high_magnitude_jerk
crossed_by_vehicle
following_lane_with_lead
stopping_at_stop_sign_without_lead
accelerating_at_crosswalk
starting_straight_traffic_light_intersection_traversal
near_multiple_bikes
stationary_in_traffic
traversing_traffic_light_intersection
stationary_at_traffic_light_without_lead
behind_bike
near_high_speed_vehicle
following_lane_without_lead
stopping_at_crosswalk
stopping_at_stop_sign_no_crosswalk
crossed_by_bike
near_pedestrian_on_crosswalk_with_ego
starting_unprotected_cross_turn
near_construction_zone_sign
on_intersection
on_stopline_traffic_light
starting_right_turn
starting_straight_stop_sign_intersection_traversal
near_barrier_on_driveable
"""

CHANGING_LANE_LIST = ["changing_lane", "changing_lane_with_lead", "changing_lane_with_trail", "changing_lane_to_right", "changing_lane_to_left"] #5
STARTING_LEFT_TURN_LIST = ["starting_left_turn"] #1
STARTING_RIGHT_TURN_LIST = ["starting_right_turn"] #1
STARTING_STRAIGHT_TRAFFIC_LIGHT_INTERSECTION_TRAVERSAL_LIST = ["starting_straight_traffic_light_intersection_traversal", "starting_straight_stop_sign_intersection_traversal"] #2
HIGH_LATERAL_ACCELERATION_LIST = ["high_lateral_acceleration", "starting_u_turn", "starting_unprotected_cross_turn", "starting_protected_cross_turn", "starting_protected_noncross_turn", "starting_unprotected_noncross_turn"] #6
LOW_MAGNITUDE_SPEED_LIST = ["low_magnitude_speed", "starting_low_speed_turn"] #2
HIGH_MAGNITUDE_SPEED_LIST = ["high_magnitude_speed", "high_magnitude_jerk", "starting_high_speed_turn"] #3
FOLLOWING_LANE_WITH_LEAD_LIST = ["following_lane_with_lead", "following_lane_with_slow_lead", "following_lane_without_lead", "medium_magnitude_speed", "accelerating_at_traffic_light", "accelerating_at_traffic_light_with_lead", \
                                "accelerating_at_stop_sign_no_crosswalk", "accelerating_at_crosswalk", "accelerating_at_stop_sign", "accelerating_at_traffic_light_without_lead"] #10
STATIONARY_IN_TRAFFIC_LIST = ["stationary_in_traffic", "stationary_at_traffic_light_with_lead", "stationary_at_crosswalk", "stationary_at_traffic_light_without_lead", "stationary"] #5
STOPPING_WITH_LEAD_LIST = ["stopping_with_lead", "stopping_at_stop_sign_without_lead", "stopping_at_stop_sign_with_lead", "stopping_at_traffic_light_without_lead", "stopping_at_traffic_light_with_lead", "stopping_at_crosswalk", \
                            "stopping_at_stop_sign_no_crosswalk"] #7
TRAVERSING_PICKUP_DROPOFF_LIST = ["traversing_pickup_dropoff", "traversing_intersection", "traversing_traffic_light_intersection", "traversing_crosswalk", "traversing_narrow_lane", "on_traffic_light_intersection", \
                                "on_intersection", "on_stopline_crosswalk", "on_all_way_stop_intersection", "on_stopline_stop_sign", "on_pickup_dropoff", "on_stopline_traffic_light"] #12
BEHIND_LOGN_VEHILCE = ["behind_long_vehicle", "behind_pedestrian_on_driveable", "behind_pedestrian_on_pickup_dropoff", "behind_bike"] #4
NEAR_MULTIPLE_VEHICLES = ["near_multiple_vehicles", "near_multiple_bikes", "near_pedestrian_on_crosswalk", "near_pedestrian_on_crosswalk_with_ego", "near_construction_zone_sign", "near_barrier_on_driveable", "near_high_speed_vehicle", \
                            "near_trafficcone_on_driveable", "near_multiple_pedestrians", "near_long_vehicle", "near_pedestrian_at_pickup_dropoff"] #11
WAITING_FOR_PEDESTRIAN_TO_CROSS = ["waiting_for_pedestrian_to_cross", "crossed_by_vehicle", "crossed_by_bike"] #3
ON = ["on_carpark"] #1