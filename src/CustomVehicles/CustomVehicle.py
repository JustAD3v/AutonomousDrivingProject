import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

# import it like this 
# to use it in env config 
"""
custom_config = {
    "other_vehicles_type": "CustomVehicles.CustomVehicle.CustomIDMVehicle",
}
"""

class CustomIDMVehicle(IDMVehicle):

    @classmethod
    def create_random(
        cls,
        road,
        speed=None,
        lane_from=None,
        lane_to=None,
        lane_id=None,
        spacing=1,
    ):
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))

        # ===== custom ======
        ego = road.vehicles[0]  # Works only if the ego is the first vehicle created
        speed = ego.speed
        
        _id = (
            lane_id
            if lane_id is not None
            else road.np_random.choice(len(road.network.graph[_from][_to]))
        )

        if road.np_random.random() < 0.6:
            lane = road.network.get_lane(ego.lane_index)
        else:
            lane = road.network.get_lane((_from, _to, _id))
        # ===== custom ======

        # if speed is None:
        #     if lane.speed_limit is not None:
        #         speed = road.np_random.uniform(
        #             0.7 * lane.speed_limit,
        #             0.8 * lane.speed_limit,
        #         )
        #     else:
        #         speed = road.np_random.uniform(
        #             IDMVehicle.DEFAULT_INITIAL_SPEEDS[0],
        #             IDMVehicle.DEFAULT_INITIAL_SPEEDS[1],
        #         )

        # Original spacing precaution
        default_spacing = 12 + speed
        offset = (
            spacing
            * default_spacing
            * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        
        # ==== custom =====
        if road.vehicles:
            # all cars positi#ons
            positions = [
                lane.local_coordinates(v.position)[0]
                for v in road.vehicles
            ]

            # safety distance between existing car and new spawned vehicle 
            distance = offset * road.np_random.uniform(0.9, 1.1)

            # choose at random to spawn beyond or behind the ego car
            if road.np_random.random() < 0.4:
                # spawn beyond the furthest car forward
                x0 = max(positions) + distance
            else:
                # behind the furthest car backward
                x0 = min(positions) - distance
        else:
            x0 = 3 * offset
        # ==== custom =====

        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v