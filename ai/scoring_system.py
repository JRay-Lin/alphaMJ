"""Mahjong scoring system for Stage 2 AI training."""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import Counter
from modules.tile import MahjongTile
from modules.winChecker import win_checker


class MahjongScoring:
    """
    Traditional Mahjong scoring system.
    
    Implements standard scoring rules including:
    - Basic hand patterns and combinations
    - Special hands (flush, all honors, etc.)
    - Multipliers for concealed hands, self-draw
    - Bonus points for dragons, winds, flowers
    """
    
    def __init__(self, rule: str = "standard"):
        """
        Initialize scoring system.
        
        Args:
            rule: Scoring rule variant ("standard", "taiwan", "japanese")
        """
        self.rule = rule
        self.win_checker = win_checker(rule)
        
        # Base scoring values
        self.base_scores = self._get_base_scores()
        self.multipliers = self._get_multipliers()
        
    def _get_base_scores(self) -> Dict[str, int]:
        """Get base scoring values for different combinations."""
        return {
            # Basic melds
            'pung_simple': 4,      # Simple tiles (2-8)
            'pung_terminal': 8,    # Terminal tiles (1, 9)
            'pung_honor': 8,       # Honor tiles (winds, dragons)
            'kong_simple': 16,     # Kong of simple tiles
            'kong_terminal': 32,   # Kong of terminal tiles
            'kong_honor': 32,      # Kong of honor tiles
            'chow': 0,             # Chows have no base score
            
            # Special hands
            'flush': 50,           # All same suit
            'mixed_flush': 30,     # Same suit + honors
            'all_pungs': 40,       # All pungs (no chows)
            'all_honors': 60,      # All honor tiles
            'all_terminals': 80,   # All terminal tiles (1, 9)
            'seven_pairs': 25,     # Seven pairs
            'thirteen_orphans': 100,  # Thirteen unique terminals + honors
            
            # Dragons and winds
            'dragon_pung': 10,     # Pung of dragons
            'wind_pung_own': 10,   # Pung of own wind
            'wind_pung_round': 10, # Pung of round wind
            'wind_pung_other': 5,  # Pung of other wind
            
            # Winning conditions
            'self_draw': 10,       # Win by self-draw
            'last_tile': 15,       # Win with last tile from wall
            'robbing_kong': 15,    # Win by robbing kong
            'concealed_hand': 20,  # All concealed (no exposed melds)
            'no_points_hand': 10,  # Hand with no basic points (must have special scoring)
        }
    
    def _get_multipliers(self) -> Dict[str, float]:
        """Get scoring multipliers."""
        return {
            'concealed': 1.5,      # Concealed vs exposed meld multiplier
            'self_draw': 1.2,      # Self-draw multiplier
            'dealer': 1.5,         # Dealer win multiplier
            'riichi': 2.0,         # Riichi declaration multiplier (if implemented)
        }
    
    def calculate_hand_score(self, hand_tiles: List[MahjongTile], 
                           exposed_sets: List[List[MahjongTile]],
                           game_context: Dict = None) -> int:
        """
        Calculate total score for a winning hand.
        
        Args:
            hand_tiles: Tiles remaining in hand
            exposed_sets: List of exposed meld sets
            game_context: Game context (wind, self_draw, etc.)
            
        Returns:
            Total hand score
        """
        if not game_context:
            game_context = {}
        
        # First verify this is a winning hand
        if not self._is_winning_hand(hand_tiles, exposed_sets):
            return 0
        
        # Calculate base score from melds
        base_score = 0
        
        # Score exposed sets
        for meld_set in exposed_sets:
            meld_score = self._score_meld(meld_set, concealed=False)
            base_score += meld_score
        
        # Score concealed melds in hand
        concealed_melds = self._extract_concealed_melds(hand_tiles, exposed_sets)
        for meld in concealed_melds:
            meld_score = self._score_meld(meld, concealed=True)
            base_score += meld_score
        
        # Add special hand bonuses
        special_bonus = self._calculate_special_bonuses(hand_tiles, exposed_sets, game_context)
        base_score += special_bonus
        
        # Apply multipliers
        multiplier = self._calculate_multipliers(hand_tiles, exposed_sets, game_context)
        final_score = int(base_score * multiplier)
        
        # Minimum score for winning hand
        return max(final_score, 10)
    
    def _is_winning_hand(self, hand_tiles: List[MahjongTile], 
                        exposed_sets: List[List[MahjongTile]]) -> bool:
        """Check if hand is a valid winning combination."""
        return self.win_checker.is_winning_hand_with_sets(hand_tiles, exposed_sets, self.rule)
    
    def _score_meld(self, meld: List[MahjongTile], concealed: bool = True) -> int:
        """Score a single meld (pung, kong, chow)."""
        if not meld:
            return 0
        
        meld_size = len(meld)
        first_tile = meld[0]
        
        # Determine meld type and tile category
        if meld_size == 4:
            # Kong
            if first_tile.type in ['風', '元']:
                score = self.base_scores['kong_honor']
            elif first_tile.number in [1, 9]:
                score = self.base_scores['kong_terminal']
            else:
                score = self.base_scores['kong_simple']
        
        elif meld_size == 3:
            # Check if pung or chow
            if all(tile.name == first_tile.name for tile in meld):
                # Pung
                if first_tile.type in ['風', '元']:
                    score = self.base_scores['pung_honor']
                elif first_tile.number in [1, 9]:
                    score = self.base_scores['pung_terminal']
                else:
                    score = self.base_scores['pung_simple']
            else:
                # Chow
                score = self.base_scores['chow']
        
        elif meld_size == 2:
            # Pair - only score if it's dragons or significant winds
            if first_tile.class_name in ['中', '發', '白']:
                score = 2  # Small bonus for dragon pair
            else:
                score = 0
        
        else:
            return 0
        
        # Apply concealed multiplier
        if concealed and score > 0:
            score = int(score * self.multipliers['concealed'])
        
        return score
    
    def _extract_concealed_melds(self, hand_tiles: List[MahjongTile],
                                exposed_sets: List[List[MahjongTile]]) -> List[List[MahjongTile]]:
        """Extract concealed melds from hand tiles."""
        if not hand_tiles:
            return []
        
        # Use win checker to identify meld structure
        # This is a simplified version - in practice, we'd need to reconstruct the winning combination
        
        # For now, return empty list and rely on exposed sets scoring
        # A full implementation would decompose the hand into its meld components
        concealed_melds = []
        
        # Simple heuristic: look for obvious pungs/kongs in hand
        tile_counts = Counter([tile.name for tile in hand_tiles])
        
        for tile_name, count in tile_counts.items():
            if count >= 3:
                # Found a pung or kong
                tile_obj = next(tile for tile in hand_tiles if tile.name == tile_name)
                if count == 4:
                    concealed_melds.append([tile_obj] * 4)  # Kong
                else:
                    concealed_melds.append([tile_obj] * 3)  # Pung
        
        return concealed_melds
    
    def _calculate_special_bonuses(self, hand_tiles: List[MahjongTile],
                                  exposed_sets: List[List[MahjongTile]],
                                  game_context: Dict) -> int:
        """Calculate bonuses for special hand patterns."""
        bonus = 0
        
        # Combine all tiles for analysis
        all_tiles = hand_tiles + [tile for meld_set in exposed_sets for tile in meld_set]
        
        # Flush bonuses
        if self._is_flush(all_tiles):
            bonus += self.base_scores['flush']
        elif self._is_mixed_flush(all_tiles):
            bonus += self.base_scores['mixed_flush']
        
        # All pungs
        if self._is_all_pungs(exposed_sets, hand_tiles):
            bonus += self.base_scores['all_pungs']
        
        # Honor/terminal bonuses
        if self._is_all_honors(all_tiles):
            bonus += self.base_scores['all_honors']
        elif self._is_all_terminals(all_tiles):
            bonus += self.base_scores['all_terminals']
        
        # Special winning conditions
        if game_context.get('self_draw', False):
            bonus += self.base_scores['self_draw']
        
        if game_context.get('last_tile', False):
            bonus += self.base_scores['last_tile']
        
        if game_context.get('robbing_kong', False):
            bonus += self.base_scores['robbing_kong']
        
        if len(exposed_sets) == 0:  # All concealed
            bonus += self.base_scores['concealed_hand']
        
        return bonus
    
    def _calculate_multipliers(self, hand_tiles: List[MahjongTile],
                              exposed_sets: List[List[MahjongTile]],
                              game_context: Dict) -> float:
        """Calculate scoring multipliers."""
        multiplier = 1.0
        
        # Self-draw multiplier
        if game_context.get('self_draw', False):
            multiplier *= self.multipliers['self_draw']
        
        # Dealer multiplier
        if game_context.get('is_dealer', False):
            multiplier *= self.multipliers['dealer']
        
        # Riichi multiplier (if implemented)
        if game_context.get('riichi', False):
            multiplier *= self.multipliers['riichi']
        
        return multiplier
    
    def _is_flush(self, tiles: List[MahjongTile]) -> bool:
        """Check if all tiles are from the same suit."""
        if not tiles:
            return False
        
        suits = set()
        for tile in tiles:
            if tile.type in ['萬', '筒', '條']:
                suits.add(tile.type)
            else:
                return False  # Honor tiles prevent flush
        
        return len(suits) == 1
    
    def _is_mixed_flush(self, tiles: List[MahjongTile]) -> bool:
        """Check if tiles are from one suit plus honors."""
        if not tiles:
            return False
        
        main_suits = set()
        has_honors = False
        
        for tile in tiles:
            if tile.type in ['萬', '筒', '條']:
                main_suits.add(tile.type)
            elif tile.type in ['風', '元']:
                has_honors = True
            else:
                return False
        
        return len(main_suits) == 1 and has_honors
    
    def _is_all_pungs(self, exposed_sets: List[List[MahjongTile]],
                     hand_tiles: List[MahjongTile]) -> bool:
        """Check if hand contains only pungs/kongs (no chows)."""
        # Check exposed sets
        for meld_set in exposed_sets:
            if len(meld_set) == 3:
                # Must be pung (all same tile)
                if not all(tile.name == meld_set[0].name for tile in meld_set):
                    return False
        
        # For hand tiles, this would require more complex analysis
        # Simplified check: assume true if no obvious chows
        return True
    
    def _is_all_honors(self, tiles: List[MahjongTile]) -> bool:
        """Check if all tiles are honor tiles (winds/dragons)."""
        return all(tile.type in ['風', '元'] for tile in tiles)
    
    def _is_all_terminals(self, tiles: List[MahjongTile]) -> bool:
        """Check if all tiles are terminals (1s and 9s)."""
        return all(tile.number in [1, 9] for tile in tiles if tile.type in ['萬', '筒', '條'])
    
    def calculate_hand_potential(self, hand_tiles: List[MahjongTile],
                               exposed_sets: List[List[MahjongTile]],
                               available_tiles: List[str] = None) -> float:
        """
        Calculate the scoring potential of a hand.
        
        This estimates how much the hand could score if completed optimally.
        Used for AI decision making in Stage 2 training.
        
        Args:
            hand_tiles: Current hand tiles
            exposed_sets: Current exposed sets
            available_tiles: Tiles still available (for probability calculation)
            
        Returns:
            Estimated scoring potential
        """
        if not hand_tiles and not exposed_sets:
            return 0.0
        
        # Base potential from existing melds
        base_potential = 0
        for meld_set in exposed_sets:
            base_potential += self._score_meld(meld_set, concealed=False)
        
        # Analyze hand for potential melds
        tile_counts = Counter([tile.name for tile in hand_tiles])
        
        # Potential pungs/kongs
        for tile_name, count in tile_counts.items():
            if count >= 2:
                tile_obj = next(tile for tile in hand_tiles if tile.name == tile_name)
                if count == 3:
                    base_potential += self._score_meld([tile_obj] * 3, concealed=True) * 0.8
                elif count == 2:
                    base_potential += self._score_meld([tile_obj] * 3, concealed=True) * 0.4
        
        # Potential special hands
        all_tiles = hand_tiles + [tile for meld_set in exposed_sets for tile in meld_set]
        
        # Flush potential
        suit_counts = Counter([tile.type for tile in all_tiles if tile.type in ['萬', '筒', '條']])
        if suit_counts:
            max_suit_ratio = max(suit_counts.values()) / len([t for t in all_tiles if t.type in ['萬', '筒', '條']])
            if max_suit_ratio > 0.6:
                base_potential += self.base_scores['flush'] * (max_suit_ratio - 0.5)
        
        # Honor tile concentration
        honor_count = len([tile for tile in all_tiles if tile.type in ['風', '元']])
        if honor_count > len(all_tiles) * 0.3:
            base_potential += self.base_scores['all_honors'] * (honor_count / len(all_tiles))
        
        return base_potential
    
    def get_scoring_summary(self, hand_tiles: List[MahjongTile],
                           exposed_sets: List[List[MahjongTile]],
                           game_context: Dict = None) -> Dict[str, any]:
        """
        Get detailed scoring breakdown for analysis.
        
        Returns:
            Dictionary with scoring details
        """
        if not game_context:
            game_context = {}
        
        summary = {
            'total_score': 0,
            'base_score': 0,
            'special_bonus': 0,
            'multiplier': 1.0,
            'meld_scores': [],
            'special_hands': [],
            'is_winning': False
        }
        
        # Check if winning
        summary['is_winning'] = self._is_winning_hand(hand_tiles, exposed_sets)
        
        if not summary['is_winning']:
            return summary
        
        # Calculate detailed breakdown
        base_score = 0
        
        # Score each exposed meld
        for i, meld_set in enumerate(exposed_sets):
            meld_score = self._score_meld(meld_set, concealed=False)
            base_score += meld_score
            summary['meld_scores'].append({
                'meld_index': i,
                'tiles': [tile.name for tile in meld_set],
                'type': self._get_meld_type(meld_set),
                'score': meld_score,
                'concealed': False
            })
        
        # Score concealed melds
        concealed_melds = self._extract_concealed_melds(hand_tiles, exposed_sets)
        for i, meld in enumerate(concealed_melds):
            meld_score = self._score_meld(meld, concealed=True)
            base_score += meld_score
            summary['meld_scores'].append({
                'meld_index': len(exposed_sets) + i,
                'tiles': [tile.name for tile in meld],
                'type': self._get_meld_type(meld),
                'score': meld_score,
                'concealed': True
            })
        
        summary['base_score'] = base_score
        
        # Special bonuses
        special_bonus = self._calculate_special_bonuses(hand_tiles, exposed_sets, game_context)
        summary['special_bonus'] = special_bonus
        
        # Identify special hands
        all_tiles = hand_tiles + [tile for meld_set in exposed_sets for tile in meld_set]
        if self._is_flush(all_tiles):
            summary['special_hands'].append('flush')
        elif self._is_mixed_flush(all_tiles):
            summary['special_hands'].append('mixed_flush')
        
        if self._is_all_honors(all_tiles):
            summary['special_hands'].append('all_honors')
        elif self._is_all_terminals(all_tiles):
            summary['special_hands'].append('all_terminals')
        
        if len(exposed_sets) == 0:
            summary['special_hands'].append('concealed_hand')
        
        # Multipliers
        multiplier = self._calculate_multipliers(hand_tiles, exposed_sets, game_context)
        summary['multiplier'] = multiplier
        
        # Final score
        summary['total_score'] = max(int((base_score + special_bonus) * multiplier), 10)
        
        return summary
    
    def _get_meld_type(self, meld: List[MahjongTile]) -> str:
        """Determine the type of a meld."""
        if not meld:
            return 'unknown'
        
        size = len(meld)
        if size == 4:
            return 'kong'
        elif size == 3:
            if all(tile.name == meld[0].name for tile in meld):
                return 'pung'
            else:
                return 'chow'
        elif size == 2:
            return 'pair'
        else:
            return 'unknown'