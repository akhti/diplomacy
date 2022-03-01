# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DiplomacyState protocol."""

from typing import Sequence, Tuple, Dict, List, Optional
import numpy as np
import typing_extensions

from fairdiplomacy import pydipcc
from fairdiplomacy.models.consts import POWERS, MAP

from dm_diplomacy.environment import observation_utils as utils
from dm_diplomacy.environment import action_utils
from dm_diplomacy.environment import mila_actions
from dm_diplomacy.environment import province_order
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary


ADJ_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
)


class DiplomacyState(typing_extensions.Protocol):
    """Diplomacy State protocol."""

    def is_terminal(self) -> bool:
        """Whether the game has ended."""
        pass

    def observation(self) -> utils.Observation:
        """Returns the current observation."""
        pass

    def legal_actions(self) -> Sequence[Sequence[int]]:
        """A list of lists of legal unit actions.

    There are 7 sub-lists, one for each power, sorted alphabetically (Austria,
    England, France, Germany, Italy, Russia, Turkey).
    The sub-list has every unit action possible in the given position, for all
    of that power's units.
    """
        pass

    def returns(self) -> np.ndarray:
        """The returns of the game. All 0s if the game is in progress."""
        pass

    def step(self, actions_per_player: Sequence[Sequence[int]]) -> None:
        """Steps the environment forward a full phase of Diplomacy.

    Args:
      actions_per_player: A list of lists of unit-actions. There are 7
        sub-lists, one per power, sorted alphabetically (Austria, England,
        France, Germany, Italy, Russia, Turkey), each sublist is all of the
        corresponding player's unit-actions for that phase.
    """
        pass


class PydipccDiplomacyState(DiplomacyState):
    def __init__(self, game: Optional[pydipcc.Game] = None):
        if game is None:
            game = pydipcc.Game()
        self.game = game

    def is_terminal(self) -> bool:
        return self.game.is_game_done

    def observation(self) -> utils.Observation:
        return utils.Observation(
            season=_encode_season(self.game),
            board=_encode_board(self.game),
            build_numbers=_encode_build_numbers(self.game),
            last_actions=_encode_last_actions(self.game),
        )

    def legal_actions(self) -> Sequence[Sequence[int]]:
        season = _encode_season_from_str(self.game.phase)
        per_power_order_ids = []
        # print(season)
        for order_strs in _get_possible_order_per_power(self.game):
            power_order_ids = [
                mila_actions.mila_action_to_action(order, season)
                for order in order_strs
            ]
            # if self.game.current_short_phase == "F1901M":
            #     print(order_strs)
            per_power_order_ids.append(power_order_ids)
        return per_power_order_ids

    def returns(self) -> np.ndarray:
        if self.game.is_game_done:
            return np.array(self.game.get_scores())
        else:
            return np.zeros(7)

    def step(self, actions_per_player: Sequence[Sequence[int]]) -> None:
        for power, order_ids, possible_order_strs in zip(
            POWERS, actions_per_player, _get_possible_order_per_power(self.game)
        ):
            possible_order_strs = frozenset(possible_order_strs)
            orders = []
            for order_id in order_ids:
                order_strs = (
                    frozenset(mila_actions.action_to_mila_actions(order_id))
                    & possible_order_strs
                )
                assert len(order_strs) == 1, order_strs
                orders.append(list(order_strs)[0])
            # print(power, orders)
            self.game.set_orders(power, orders)
        self.game.process()


def _get_possible_order_per_power(game: pydipcc.Game) -> List[List[str]]:
    order_vocab = get_order_vocabulary()
    loc2orders = game.get_all_possible_orders()
    per_power_order_strs = []
    power2locs = game.get_orderable_locations()
    for power in POWERS:
        locs = power2locs[power]
        power_order_strs = sum([loc2orders[loc] for loc in locs], [])
        power_order_strs = [x for x in power_order_strs if x in order_vocab]
        per_power_order_strs.append(power_order_strs)
    return per_power_order_strs


def _encode_season(game: pydipcc.Game) -> utils.Season:
    assert not game.is_game_done
    return _encode_season_from_str(game.phase)


def _encode_season_from_str(phase: str) -> utils.Season:
    season, _, action_type = phase.split()
    mapping: Dict[Tuple[str, str], utils.Season] = {
        ("SPRING", "MOVEMENT"): utils.Season.SPRING_MOVES,
        ("FALL", "MOVEMENT"): utils.Season.AUTUMN_MOVES,
        ("SPRING", "RETREATS"): utils.Season.SPRING_RETREATS,
        ("FALL", "RETREATS"): utils.Season.AUTUMN_RETREATS,
        ("WINTER", "ADJUSTMENTS"): utils.Season.BUILDS,
    }
    return mapping[(season, action_type)]


def _encode_board(game: pydipcc.Game) -> np.ndarray:
    """An array of shape (observation_utils.NUM_AREAS, utils.PROVINCE_VECTOR_LENGTH).
    
    The areas are ordered as in bicoastal_map.mdf. The vector representing a single area is, in order:

            3 flags representing the presence of an army, a fleet or an empty province respectively
            7 flags representing the owner of the unit, plus an 8th that is true if there is no such unit
            1 flag representing whether a unit can be built in the province
            1 flag representing whether a unit can be removed from the province
            3 flags representing the existence of a dislodged army or fleet, or no dislodged unit
            7 flags representing the owner of the dislodged unit, plus an 8th that is true if there is no such unit
            3 flags for land-sea-coast
            7 flags representing the owner of the supply centre in the province, plus an 8th representing an unowned supply centre. The 8th flag is false if there is no SC in the area
    """
    assert not game.is_game_done
    phase_data = game.get_phase_data().state

    # prepare some dicts in [loc -> data] format
    province2unit_type = {}
    province2unit_owner = {}
    for power, power_units in phase_data["units"].items():
        for unit in power_units:
            unit_type, unit_loc = unit.split()
            if "*" in unit_type:
                continue
            province2unit_type[unit_loc] = unit_type.strip("*")
            province2unit_owner[unit_loc] = power
            # Count this unit for non-coastal version of the locaiton as well
            unit_loc_base = unit_loc.split("/")[0]
            province2unit_type[unit_loc_base] = unit_type.strip("*")
            province2unit_owner[unit_loc_base] = power
    province2can_builld = {}
    province2can_remove = {}
    for power, build_data in phase_data["builds"].items():
        if build_data["count"] > 0:
            for loc in build_data["homes"]:
                province2can_builld[loc] = power
        elif build_data["count"] < 0:
            for unit in phase_data["units"][power]:
                _, unit_loc = unit.split()
                province2can_remove[unit_loc] = power

    province2retreat_unit = {}
    province2retreat_unit_owner = {}
    for power, retreat_data in phase_data["retreats"].items():
        for unit in retreat_data:
            unit_type, unit_loc = unit.split()
            unit_loc = unit_loc.split("/")[0]
            province2retreat_unit[unit_loc] = unit_type
            province2retreat_unit_owner[unit_loc] = power

    province_sc2owner = {}
    for power, scs in phase_data["centers"].items():
        for sc in scs:
            province_sc2owner[sc] = power

    def dm_loc_to_mila_loc(loc):
        LOC_MAPPINGS = {"ECH": "ENG", "GOB": "BOT", "GOL": "LYO"}
        return LOC_MAPPINGS.get(loc[:3], loc[:3]) + loc[3:]

    # if province2can_remove:
    #     print(phase_data["builds"], province2can_remove)

    # Write into an array
    provinces_id_pairs = list(
        (dm_loc_to_mila_loc(prov), pid)
        for prov, pid in province_order.province_name_to_id(
            province_order.MapMDF.BICOASTAL_MAP
        ).items()
    )
    baseprovinces_id_pairs = [
        (province.split("/")[0], pid) for province, pid in provinces_id_pairs
    ]
    board = np.zeros((utils.NUM_AREAS, utils.PROVINCE_VECTOR_LENGTH), np.float32)
    assert len(board) == len(provinces_id_pairs)
    offset = 0

    for province, pid in provinces_id_pairs:
        local_offset = {"A": 0, "F": 1, None: 2}[province2unit_type.get(province)]
        board[pid, offset + local_offset] = 1
    offset += 3

    # print(province2unit_owner)
    for province, pid in provinces_id_pairs:
        if province in province2unit_owner:
            local_offset = POWERS.index(province2unit_owner[province])
        else:
            local_offset = 7
        board[pid, offset + local_offset] = 1
    offset += 8

    for province_base, pid in baseprovinces_id_pairs:
        if province_base in province2can_builld:
            board[pid, offset] = 1
    offset += 1

    for province, pid in provinces_id_pairs:
        if province in province2can_remove:
            board[pid, offset] = 1
    offset += 1

    for province_base, pid in baseprovinces_id_pairs:
        local_offset = {"A": 0, "F": 1, None: 2}[
            province2retreat_unit.get(province_base)
        ]
        board[pid, offset + local_offset] = 1
    offset += 3

    for province, pid in provinces_id_pairs:
        if province in province2retreat_unit_owner:
            local_offset = POWERS.index(province2retreat_unit_owner[province])
        else:
            local_offset = 7
        board[pid, offset + local_offset] = 1
    offset += 8

    board[:, offset : offset + 3] = ADJ_MATRIX
    offset += 3

    for province_base, pid in baseprovinces_id_pairs:
        if province_base in province_sc2owner:
            local_offset = POWERS.index(province_sc2owner[province_base])
        elif province_base in MAP.scs:
            local_offset = 7
        else:
            continue
        board[pid, offset + local_offset] = 1
    offset += 8

    assert offset == utils.PROVINCE_VECTOR_LENGTH, (
        offset,
        utils.PROVINCE_VECTOR_LENGTH,
    )

    return board


def _encode_build_numbers(game: pydipcc.Game) -> List[int]:
    assert not game.is_game_done
    if game.phase_type == "A":
        phase_data = game.get_phase_data().state
        numbers = [phase_data["builds"][power]["count"] for power in POWERS]
    else:
        # So, this is a weird thing. In non-adjustment phases we use retreats
        # from the last adjustment phase.
        adj_phases = [x for x in game.get_phase_history() if x.name.endswith("A")]
        if adj_phases:
            phase_data = adj_phases[-1].state
            numbers = [min(0, phase_data["builds"][power]["count"]) for power in POWERS]
        else:
            numbers = [0] * 7
    return numbers


def _encode_last_actions(game: pydipcc.Game) -> List:
    history = game.get_phase_history()
    if not history:
        return []
    season = _encode_season_from_str(
        game.rolled_back_to_phase_start(history[-1].name).phase
    )
    order_ids = []
    for orders in history[-1].orders.values():
        for order in orders:
            order_ids.append(mila_actions.mila_action_to_action(order, season))
    return sorted(order_ids, key=action_utils.ordered_province)
