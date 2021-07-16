### Adversarial Surprise
### Arnaud Fickinger, 2021

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos

class MultiRoomABEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        minRoomSize = 4,
        maxRoomSize=10,
        max_steps=100,
        proba_dark = 0.5,
        proba_ball = 0.25,
        proba_floor = 0.25,
        seed=1,
        proportion_obst_floor = 0.5,
        proportion_obst_ball = 0.5,
        view_size = 7
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize
        self.minRoomSize = minRoomSize
        self.proportion_obst_ball = proportion_obst_ball
        self.proportion_obst_floor = proportion_obst_floor

        self.room_visited = set()
        self.room_visited_episode = set()

        self.rooms = []
        self.proba_dark = proba_dark
        self.proba_ball = proba_ball
        self.proba_floor = proba_floor

        super(MultiRoomABEnv, self).__init__(
            grid_size=25,
            max_steps=max_steps,
            seed=seed,
            agent_view_size = view_size
            # self.maxNumRooms * 20,
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        # print("rooms")

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=self.minRoomSize,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList
        self.room_types = []
        self.obstacles = {}
        self.noisy_floors = {}
        self.switches = {}
        self.boxes = {}
        self.moving_balls = {}
        self.moving_floor = {}

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            # print(f'room {idx}')

            topX, topY = room.top
            sizeX, sizeY = room.size

            room_type = self.np_random.choice(sorted([0,1,2]), p=[self.proba_dark, self.proba_ball, self.proba_floor])

            self.room_types.append(room_type)
            # import pdb; pdb.set_trace()

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            if room_type == 1:  # balls
                switch = Switch(color=self._rand_elem(COLOR_NAMES), idx=idx)
                try:
                    self.place_obj(switch,
                                   top=room.top,
                                   size=room.size,
                                   max_tries=10*(sizeX-2)*(sizeY-2))
                except:
                    pass
                    # print("not enough space for switch")
                obstacle = []
                for _ in range(min(int(self.proportion_obst_ball*(sizeX-2)*(sizeY-2)), (sizeX-2)*(sizeY-2)-2)):
                    obstacle.append(Ball(color=self._rand_elem(COLOR_NAMES)))
                    try:
                        self.place_obj(obstacle[-1],
                                       top=room.top,
                                       size=room.size,
                                       max_tries=10*(sizeX-2)*(sizeY-2))
                    except:
                        # print("not enough space for another ball")
                        break
                self.obstacles[idx] = obstacle


                self.switches[idx] = switch
                self.moving_balls[idx]=True
            elif room_type == 2: #floor
                floors = []
                box = Box(color=self._rand_elem(COLOR_NAMES), idx=idx)
                try:
                    self.place_obj(box,
                                   top=room.top,
                                   size=room.size,
                                   max_tries=10*(sizeX-2)*(sizeY-2))
                except:
                    pass
                    # print("not enough space for box")
                for _ in range(min(int(self.proportion_obst_floor*(sizeX-2)*(sizeY-2)), (sizeX-2)*(sizeY-2)-1)):
                    floors.append(Floor(color=self._rand_elem(COLOR_NAMES)))
                    try:
                        self.place_obj(floors[-1],
                                       top=room.top,
                                       size=room.size,
                                       max_tries=10*(sizeX-2)*(sizeY-2))
                    except:
                        # print("not enough space for floor")
                        break
                self.noisy_floors[idx] = floors

                self.boxes[idx] = box
                self.moving_floor[idx] = True

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        # print("agent")
        self.place_agent(roomList[0].top, roomList[0].size)
        # print("ok")

        # Place the final goal in the last room
        # self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def current_room(self):
        for idx, room in enumerate(self.rooms):
            topX, topY = room.top
            sizeX, sizeY = room.size
            if self.agent_pos[0]>=topX and self.agent_pos[0]<topX+sizeX and self.agent_pos[1]>=topY and self.agent_pos[1]<topY+sizeY:
                return idx
        assert False

    def reset(self):
        super().reset()
        assert self.step_count == 0

        self.room_visited_episode.clear()

        obs = self.gen_obs()

        return obs

    def step(self, action):
        # print("step")
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            self.step_count = 0
        else:
            done = False

        for idx in self.moving_balls:
            if self.moving_balls[idx]:
                topX, topY = self.rooms[idx].top
                sizeX, sizeY = self.rooms[idx].size
                for obst in self.obstacles[idx]:
                    if obst is None:
                        continue
                    old_pos = obst.cur_pos
                    if old_pos is None:
                        continue
                    a = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                    b = self._rand_elem(sorted([0,1,2,3]))
                    pert = a[b]
                    if old_pos[0] + pert[0] < topX or old_pos[1] + pert[1] < topY or old_pos[0] + pert[0] > topX+sizeX or old_pos[1] + pert[1] > topY+sizeY:
                        continue
                    try:
                        self.place_obj(obst, top=([old_pos[i] + pert[i] for i in range(2)]), size=(1, 1),
                                       max_tries=3)
                        self.grid.set(*old_pos, None)
                    except:
                        pass

        for idx in self.moving_floor:
            if self.moving_floor[idx]:
                for floor in self.noisy_floors[idx]:
                    floor.color = self._rand_elem(COLOR_NAMES)

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        current_room = self.current_room()

        goal_reached = False

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
                current_room = self.current_room()
            if fwd_cell != None and fwd_cell.type == 'goal':
                # done = True
                goal_reached = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'switch':
                self.agent_pos = fwd_pos
                current_room = self.current_room()
                self.moving_balls[fwd_cell.idx]=False

        # Pick up an object
        elif action == self.actions.pickup:
            pass

        # Drop an object
        elif action == self.actions.drop:
            pass

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                if fwd_cell.type == 'door':
                    fwd_cell.toggle(self, fwd_pos)
                elif fwd_cell.type == 'box':
                    fwd_cell.toggle(self, fwd_pos)
                    self.moving_floor[fwd_cell.idx]=False

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        obs = self.gen_obs()

        reward = 0

        info = {}

        self.room_visited_episode.add(current_room)
        self.room_visited.add(current_room)

        info["agent_is_still"] = int(action in [3, 4, 5, 6])
        info["x_position"] = self.agent_pos[0]
        info["y_position"] = self.agent_pos[1]
        info["direction"] = self.agent_dir
        info["goal_reached"] = int(goal_reached)
        info["current_room"] = current_room
        info["nb_rooms_visited_in_episode"] = len(self.room_visited_episode)
        info["nb_rooms_visited_in_life"] = len(self.room_visited)
        info["entropy_reduced"] = 0 if len(self.moving_balls)+len(self.moving_floor)==0 else (len([i for i in self.moving_balls if not self.moving_balls[i]])+len([i for i in self.moving_floor if not self.moving_floor[i]]))/(len(self.moving_balls)+len(self.moving_floor))
        entropy_reduced_in_current_room = 0
        if current_room in self.moving_balls:
            if self.moving_balls[current_room]:
                entropy_reduced_in_current_room = -1
            else:
                entropy_reduced_in_current_room = 1
        if current_room in self.moving_floor:
            if self.moving_floor[current_room]:
                entropy_reduced_in_current_room = -1
            else:
                entropy_reduced_in_current_room = 1
        info["entropy_in_current_room"] = entropy_reduced_in_current_room
        entropy_reduced_in_current_room_penalty = 0
        if current_room in self.moving_balls:
            if self.moving_balls[current_room]:
                entropy_reduced_in_current_room_penalty = -1
        if current_room in self.moving_floor:
            if self.moving_floor[current_room]:
                entropy_reduced_in_current_room_penalty = -1
        info["entropy_reduced_in_current_room_penalty"] = entropy_reduced_in_current_room_penalty

        return obs, reward, done, info


class MultiRoomABEnvN2S4(MultiRoomABEnv):
    def __init__(self, max_steps=100,proba_dark = 0.5,
        proba_ball = 0.25,
        proba_floor = 0.25):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=4,
            max_steps=max_steps,
            proba_dark=proba_dark,
            proba_ball=proba_ball,
            proba_floor=proba_floor
        )

class MultiRoomABEnvN2S6(MultiRoomABEnv):
    def __init__(self, max_steps=100,proba_dark = 0.5,
        proba_ball = 0.25,
        proba_floor = 0.25):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=4,
            maxRoomSize=6,
            max_steps=max_steps,
            proba_dark=proba_dark,
            proba_ball=proba_ball,
            proba_floor=proba_floor
        )

class MultiRoomABEnvN4S5(MultiRoomABEnv):
    def __init__(self, max_steps=100,proba_dark = 0.5,
        proba_ball = 0.25,
        proba_floor = 0.25):
        super().__init__(
            minNumRooms=4,
            maxNumRooms=4,
            maxRoomSize=5,
            max_steps=max_steps,
            proba_dark=proba_dark,
            proba_ball=proba_ball,
            proba_floor=proba_floor
        )

class MultiRoomABEnvN6(MultiRoomABEnv):
    def __init__(self, max_steps=100,proba_dark = 0.5,
        proba_ball = 0.25,
        proba_floor = 0.25):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6,
            max_steps=max_steps,
            proba_dark=proba_dark,
            proba_ball=proba_ball,
            proba_floor=proba_floor
        )

class MultiRoomABEnvC(MultiRoomABEnv):
    def __init__(self, max_steps=100,
                proba_dark = 0.5,
                proba_ball = 0.25,
                proba_floor = 0.25,
                minNumRooms=4,
                maxNumRooms=4,
                maxRoomSize=6,
                minRoomSize=4,
                view_size = 7,
                seed=1,
                 proportion_obst_floor=0.5,
                 proportion_obst_ball=0.5
                 ):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=maxRoomSize,
            minRoomSize=minRoomSize,
            max_steps=max_steps,
            proba_dark=proba_dark,
            proba_ball=proba_ball,
            proba_floor=proba_floor,
            view_size=view_size,
            seed=seed,
            proportion_obst_floor=proportion_obst_floor,
            proportion_obst_ball=proportion_obst_ball
        )

register(
    id='MiniGrid-MultiRoomAB-N2-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomABEnvN2S4'
)

register(
    id='MiniGrid-MultiRoomAB-N2-S6-v0',
    entry_point='gym_minigrid.envs:MultiRoomABEnvN2S6'
)

register(
    id='MiniGrid-MultiRoomAB-N4-S5-v0',
    entry_point='gym_minigrid.envs:MultiRoomABEnvN4S5'
)

register(
    id='MiniGrid-MultiRoomAB-N6-v0',
    entry_point='gym_minigrid.envs:MultiRoomABEnvN6'
)

register(
    id='MiniGrid-MultiRoomAB-v0',
    entry_point='gym_minigrid.envs:MultiRoomABEnvC'
)
