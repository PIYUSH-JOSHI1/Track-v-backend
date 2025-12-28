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