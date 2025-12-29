import numpy as np
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

@dataclass
class VehicleData:
    """Enhanced vehicle data structure"""
    vehicle_id: int
    vehicle_type: str  # car, truck, bus, bike, emergency
    position: Tuple[int, int]
    speed: float
    queue_position: int
    wait_time: float
    priority_level: int  # 0=normal, 1=high, 2=emergency

@dataclass
class LaneMetrics:
    """Comprehensive lane analysis"""
    vehicle_count: int
    queue_length: float
    average_speed: float
    saturation_level: float
    discharge_rate: float
    arrival_rate: float
    wait_time_avg: float
    bottleneck_severity: float

class SignalPhase:
    """Represents a traffic signal phase for 4-way junction"""
    NORTH_SOUTH = 0  # Phase 0: North and South lanes are GREEN
    EAST_WEST = 1    # Phase 1: East and West lanes are GREEN
    
    # Signal states
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

class IntelligentTrafficOptimizer:
    def __init__(self):
        # Enhanced constants based on traffic engineering principles
        self.VEHICLE_CAPACITIES = {
            'car': 1.0,
            'truck': 2.5,     # Takes more space
            'bus': 3.0,       # Large vehicle
            'bike': 0.3,      # Small footprint
            'emergency': 1.0  # Same as car but highest priority
        }
        
        self.DISCHARGE_RATES = {
            'car': 2.1,       # vehicles per second during green
            'truck': 1.5,
            'bus': 1.3,
            'bike': 2.5,
            'emergency': 2.1
        }
        
        # ===== NEW: 4-WAY JUNCTION PHASE LOGIC =====
        # Lane mapping for 4-way intersection
        # Lane 0 = North, Lane 1 = East, Lane 2 = South, Lane 3 = West
        self.LANE_PHASES = {
            0: SignalPhase.NORTH_SOUTH,  # North lane
            1: SignalPhase.EAST_WEST,    # East lane
            2: SignalPhase.NORTH_SOUTH,  # South lane
            3: SignalPhase.EAST_WEST     # West lane
        }
        
        # Minimum and maximum green times
        self.MIN_GREEN_TIME = 15  # seconds
        self.MAX_GREEN_TIME = 60  # seconds
        self.YELLOW_TIME = 3      # seconds
        
        # Current phase state
        self.current_phase = SignalPhase.NORTH_SOUTH
        self.phase_start_time = time.time()
        self.phase_green_time = self.MIN_GREEN_TIME
        
        # Traffic engineering standards
        self.MIN_GREEN = 7     # Minimum green time (seconds)
        self.MAX_GREEN = 120   # Maximum green time
        self.YELLOW_TIME = 4   # Standard yellow duration
        self.ALL_RED_TIME = 2  # Clearance time
        
        # System parameters
        self.SATURATION_THRESHOLD = 0.85  # 85% capacity
        self.BOTTLENECK_THRESHOLD = 0.9   # 90% indicates bottleneck
        
        # Historical data for pattern learning
        self.traffic_history = defaultdict(lambda: deque(maxlen=100))
        self.bottleneck_history = defaultdict(lambda: deque(maxlen=50))
        
        # Multi-objective optimization weights
        self.weights = {
            'throughput': 0.4,     # Maximize vehicles processed
            'waiting_time': 0.3,   # Minimize average wait
            'queue_balance': 0.2,  # Balance between lanes
            'emergency': 0.1       # Emergency vehicle priority
        }
    
    def get_signal_state(self, lane_id: int) -> str:
        """
        Determine the signal state for a specific lane based on current phase
        
        Args:
            lane_id: 0=North, 1=East, 2=South, 3=West
        
        Returns:
            "GREEN", "YELLOW", or "RED"
        """
        lane_phase = self.LANE_PHASES[lane_id]
        elapsed_time = time.time() - self.phase_start_time
        
        # Check for emergency override
        if self.current_phase == SignalPhase.NORTH_SOUTH:
            phase_is_green = (lane_phase == SignalPhase.NORTH_SOUTH)
        else:
            phase_is_green = (lane_phase == SignalPhase.EAST_WEST)
        
        if phase_is_green:
            if elapsed_time < self.phase_green_time:
                return SignalPhase.GREEN
            elif elapsed_time < self.phase_green_time + self.YELLOW_TIME:
                return SignalPhase.YELLOW
            else:
                return SignalPhase.RED
        else:
            return SignalPhase.RED
    
    def get_green_time_for_lane(self, lane_id: int) -> float:
        """Get remaining green time for a lane"""
        state = self.get_signal_state(lane_id)
        if state == SignalPhase.GREEN:
            elapsed = time.time() - self.phase_start_time
            return max(0, self.phase_green_time - elapsed)
        return 0
    
    def calculate_optimal_phase_duration(self, lane_metrics: Dict[int, LaneMetrics]) -> float:
        """
        Calculate optimal green time for current phase based on vehicle counts
        
        Uses Webster's formula adapted for phase-based control
        """
        # Get vehicle counts for lanes in current phase
        if self.current_phase == SignalPhase.NORTH_SOUTH:
            phase_lanes = [0, 2]  # North and South
        else:
            phase_lanes = [1, 3]  # East and West
        
        # Calculate total saturation for phase
        total_saturation = sum(
            lane_metrics[lane].saturation_level 
            for lane in phase_lanes 
            if lane in lane_metrics
        ) / len(phase_lanes)
        
        # Webster's formula for green time
        # G = (1.5 * L + 5) / (1 - Y) where L is lost time, Y is saturation
        lost_time = self.ALL_RED_TIME  # clearance time
        saturation = min(total_saturation, 0.95)  # Cap at 95%
        
        if saturation < 0.95:
            green_time = (1.5 * lost_time + 5) / (1 - saturation)
        else:
            green_time = self.MAX_GREEN
        
        # Constrain to min/max
        return max(self.MIN_GREEN, min(green_time, self.MAX_GREEN))
    
    def update_phase(self, lane_metrics: Dict[int, LaneMetrics]) -> None:
        """Update traffic signal phase based on vehicle density"""
        elapsed_time = time.time() - self.phase_start_time
        
        # Check if it's time to switch phase
        if elapsed_time >= self.phase_green_time + self.YELLOW_TIME + self.ALL_RED_TIME:
            # Switch to next phase
            if self.current_phase == SignalPhase.NORTH_SOUTH:
                self.current_phase = SignalPhase.EAST_WEST
            else:
                self.current_phase = SignalPhase.NORTH_SOUTH
            
            # Calculate new green time for next phase
            self.phase_green_time = self.calculate_optimal_phase_duration(lane_metrics)
            self.phase_start_time = time.time()
    
    def get_all_signal_states(self, lane_metrics: Dict[int, LaneMetrics]) -> Dict[int, dict]:
        """
        Get signal state for all 4 lanes with timing info
        
        Returns:
            {
                0: {"state": "GREEN", "duration": 25.3, "phase": "NORTH_SOUTH"},
                1: {"state": "RED", "duration": 0, "phase": "EAST_WEST"},
                ...
            }
        """
        self.update_phase(lane_metrics)
        
        signals = {}
        for lane_id in range(4):
            signals[lane_id] = {
                "state": self.get_signal_state(lane_id),
                "duration": self.get_green_time_for_lane(lane_id),
                "phase": "NORTH_SOUTH" if self.LANE_PHASES[lane_id] == SignalPhase.NORTH_SOUTH else "EAST_WEST"
            }
        
        return signals

    def analyze_lane_conditions(self, vehicles: List[VehicleData]) -> LaneMetrics:
        """Comprehensive lane analysis using traffic engineering principles"""
        
        if not vehicles:
            return LaneMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate traffic metrics
        vehicle_count = len(vehicles)
        
        # Queue length estimation (based on vehicle spacing and position)
        queue_vehicles = [v for v in vehicles if v.speed < 2.0]  # Nearly stopped vehicles
        queue_length = len(queue_vehicles) * 7  # Average vehicle length + spacing
        
        # Average speed calculation
        avg_speed = np.mean([v.speed for v in vehicles]) if vehicles else 0
        
        # Capacity-based saturation (vehicles per hour capacity)
        lane_capacity = 1800  # Standard lane capacity (vehicles/hour)
        current_flow = vehicle_count * 3600 / 60  # Convert to vehicles/hour
        saturation_level = min(current_flow / lane_capacity, 1.0)
        
        # Discharge rate calculation (vehicles that can clear per green cycle)
        total_pcu = sum(self.VEHICLE_CAPACITIES.get(v.vehicle_type, 1.0) for v in vehicles)
        avg_discharge_rate = np.mean([self.DISCHARGE_RATES.get(v.vehicle_type, 2.0) for v in vehicles])
        
        # Arrival rate (vehicles arriving per second)
        recent_arrivals = len([v for v in vehicles if v.wait_time < 10])
        arrival_rate = recent_arrivals / 10.0
        
        # Average waiting time
        wait_time_avg = np.mean([v.wait_time for v in vehicles]) if vehicles else 0
        
        # Bottleneck severity calculation
        if saturation_level > self.SATURATION_THRESHOLD:
            bottleneck_severity = min((saturation_level - self.SATURATION_THRESHOLD) / 
                                    (1 - self.SATURATION_THRESHOLD), 1.0)
        else:
            bottleneck_severity = 0
        
        return LaneMetrics(
            vehicle_count=vehicle_count,
            queue_length=queue_length,
            average_speed=avg_speed,
            saturation_level=saturation_level,
            discharge_rate=avg_discharge_rate,
            arrival_rate=arrival_rate,
            wait_time_avg=wait_time_avg,
            bottleneck_severity=bottleneck_severity
        )

    def detect_bottleneck_situations(self, all_lanes: Dict[str, LaneMetrics]) -> Dict[str, str]:
        """Advanced bottleneck detection with root cause analysis"""
        
        bottlenecks = {}
        
        for lane_id, metrics in all_lanes.items():
            
            # Type 1: Capacity bottleneck
            if metrics.saturation_level > self.BOTTLENECK_THRESHOLD:
                bottlenecks[lane_id] = "CAPACITY_BOTTLENECK"
            
            # Type 2: Speed bottleneck (slow discharge)
            elif metrics.average_speed < 5 and metrics.queue_length > 50:
                bottlenecks[lane_id] = "DISCHARGE_BOTTLENECK"
            
            # Type 3: Spillback bottleneck (queue backing up)
            elif metrics.queue_length > 100:  # meters
                bottlenecks[lane_id] = "SPILLBACK_BOTTLENECK"
            
            # Type 4: Arrival surge bottleneck
            elif metrics.arrival_rate > metrics.discharge_rate * 1.5:
                bottlenecks[lane_id] = "ARRIVAL_SURGE"
        
        # Cross-lane analysis
        max_saturation = max([m.saturation_level for m in all_lanes.values()], default=0)
        avg_saturation = np.mean([m.saturation_level for m in all_lanes.values()])
        
        # System-wide bottleneck
        if max_saturation > 0.95 and avg_saturation > 0.8:
            bottlenecks["SYSTEM"] = "NETWORK_CONGESTION"
        
        return bottlenecks

    def calculate_optimal_green_time(self, 
                                   current_lane: LaneMetrics,
                                   other_lanes: List[LaneMetrics],
                                   bottlenecks: Dict[str, str],
                                   emergency_present: bool = False) -> int:
        """
        Multi-objective optimization for green time calculation
        Based on Webster's formula + modern adaptive control
        """
        
        # Emergency override
        if emergency_present:
            return max(15, self.MIN_GREEN)  # Quick clearance for emergency
        
        # Base calculation using Webster's optimal cycle formula
        # G = (q/s) * (C - L) where q=arrival rate, s=saturation flow, C=cycle, L=lost time
        
        # Saturation flow rate (vehicles/second during green)
        saturation_flow = current_lane.discharge_rate
        
        # Arrival demand
        demand = current_lane.arrival_rate
        
        # Lost time per cycle (yellow + all-red)
        lost_time = self.YELLOW_TIME + self.ALL_RED_TIME
        
        # Base green time calculation
        if saturation_flow > 0:
            # Webster's formula adapted
            cycle_length = 90  # Typical cycle length
            base_green = (demand / saturation_flow) * (cycle_length - lost_time)
        else:
            base_green = self.MIN_GREEN
        
        # Adjustment factors
        adjustments = []
        
        # 1. Queue clearance adjustment
        if current_lane.queue_length > 0:
            queue_clearance_time = current_lane.queue_length / 7 / saturation_flow
            adjustments.append(queue_clearance_time * 0.8)  # 80% of theoretical time
        
        # 2. Bottleneck compensation
        if "CAPACITY_BOTTLENECK" in bottlenecks.values():
            adjustments.append(base_green * 0.3)  # 30% increase
        elif "SPILLBACK_BOTTLENECK" in bottlenecks.values():
            adjustments.append(base_green * 0.5)  # 50% increase for spillback
        
        # 3. Cross-lane equity (balance with other lanes)
        other_demands = [lane.arrival_rate for lane in other_lanes]
        if other_demands:
            avg_other_demand = np.mean(other_demands)
            if current_lane.arrival_rate > avg_other_demand * 1.5:
                adjustments.append(base_green * 0.2)  # 20% increase for high demand
        
        # 4. Historical pattern adjustment
        current_hour = time.localtime().tm_hour
        if current_hour in [8, 9, 17, 18, 19]:  # Rush hours
            adjustments.append(base_green * 0.15)  # 15% increase during rush
        
        # Apply adjustments
        final_green = base_green + sum(adjustments)
        
        # Constraints
        final_green = max(self.MIN_GREEN, min(final_green, self.MAX_GREEN))
        
        # Round to practical values
        return int(final_green)

    def bottleneck_mitigation_strategy(self, 
                                     bottlenecks: Dict[str, str],
                                     all_lanes: Dict[str, LaneMetrics]) -> Dict[str, str]:
        """Generate specific strategies for different bottleneck types"""
        
        strategies = {}
        
        for lane_id, bottleneck_type in bottlenecks.items():
            
            if bottleneck_type == "CAPACITY_BOTTLENECK":
                strategies[lane_id] = "INCREASE_GREEN_TIME"
            
            elif bottleneck_type == "DISCHARGE_BOTTLENECK":
                strategies[lane_id] = "OPTIMIZE_SIGNAL_COORDINATION"
            
            elif bottleneck_type == "SPILLBACK_BOTTLENECK":
                strategies[lane_id] = "EMERGENCY_CYCLE_EXTENSION"
            
            elif bottleneck_type == "ARRIVAL_SURGE":
                strategies[lane_id] = "UPSTREAM_SIGNAL_COORDINATION"
            
            elif bottleneck_type == "NETWORK_CONGESTION":
                strategies[lane_id] = "SYSTEM_WIDE_OPTIMIZATION"
        
        return strategies

    def adaptive_signal_control(self,
                              lane_data: Dict[str, List[VehicleData]],
                              current_signal_state: str,
                              time_in_current_state: int) -> Tuple[str, int, Dict[str, str]]:
        """
        Main control logic combining all optimization strategies
        """
        
        # Analyze all lanes
        lane_metrics = {}
        emergency_detected = False
        
        for lane_id, vehicles in lane_data.items():
            lane_metrics[lane_id] = self.analyze_lane_conditions(vehicles)
            
            # Check for emergency vehicles
            if any(v.vehicle_type == 'emergency' for v in vehicles):
                emergency_detected = True
        
        # Detect bottlenecks
        bottlenecks = self.detect_bottleneck_situations(lane_metrics)
        
        # Generate mitigation strategies
        strategies = self.bottleneck_mitigation_strategy(bottlenecks, lane_metrics)
        
        # Determine optimal signal timing
        if current_signal_state == "GREEN":
            current_lane_id = "current"  # This would be determined by signal phase
            
            if current_lane_id in lane_metrics:
                current_metrics = lane_metrics[current_lane_id]
                other_metrics = [m for k, m in lane_metrics.items() if k != current_lane_id]
                
                optimal_green = self.calculate_optimal_green_time(
                    current_metrics, other_metrics, bottlenecks, emergency_detected
                )
                
                # Decide on signal transition
                if emergency_detected:
                    return "EMERGENCY_OVERRIDE", 15, strategies
                elif time_in_current_state >= optimal_green:
                    return "YELLOW", self.YELLOW_TIME, strategies
                else:
                    return "GREEN", optimal_green - time_in_current_state, strategies
            
        elif current_signal_state == "YELLOW":
            if time_in_current_state >= self.YELLOW_TIME:
                return "RED", self.ALL_RED_TIME, strategies
                
        elif current_signal_state == "RED":
            if time_in_current_state >= self.ALL_RED_TIME:
                # Determine next green phase based on demand
                max_demand_lane = max(lane_metrics.keys(), 
                                    key=lambda k: lane_metrics[k].arrival_rate)
                return "GREEN", self.MIN_GREEN, strategies
        
        return current_signal_state, time_in_current_state, strategies

# Example usage function
def create_enhanced_traffic_detector():
    """Factory function to create optimized detector"""
    
    class EnhancedVehicleDetector:
        def __init__(self):
            self.optimizer = IntelligentTrafficOptimizer()
            self.current_state = "GREEN"
            self.state_start_time = time.time()
            self.vehicles_data = defaultdict(list)
            
        def process_frame_with_optimization(self, frame, lane_id="lane_0"):
            """Enhanced processing with real-world optimization"""
            
            # ... existing YOLO detection code ...
            
            # Convert to enhanced vehicle data
            enhanced_vehicles = []
            # This would be populated from your existing detection results
            
            # Apply intelligent signal control
            time_in_state = time.time() - self.state_start_time
            
            new_state, duration, strategies = self.optimizer.adaptive_signal_control(
                {lane_id: enhanced_vehicles}, 
                self.current_state, 
                int(time_in_state)
            )
            
            if new_state != self.current_state:
                self.current_state = new_state
                self.state_start_time = time.time()
            
            return frame, len(enhanced_vehicles), duration, new_state, strategies
    
    return EnhancedVehicleDetector()
