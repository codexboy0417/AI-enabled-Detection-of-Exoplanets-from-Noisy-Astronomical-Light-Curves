from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from ..schemas import ScenarioInfo

router = APIRouter(prefix="/api/v1", tags=["scenarios"])


# Built-in scenarios matching the frontend simulator
SCENARIOS = {
    "kepler_186f": {
        "id": "kepler_186f",
        "name": "Kepler-186f (Habitable Zone Exoplanet)",
        "type": "Confirmed Exoplanet",
        "description": "An Earth-sized planet orbiting in the habitable zone of a cool M-dwarf star. Transit signal is tiny (~0.08% dip) and heavily obscured by noise.",
        "metadata": {
            "status": "Transit Detected",
            "confidence": 0.98,
            "period": 12.94,
            "depth": 0.08,
            "duration": 3.46,
            "radius": 1.17,
            "habitability": "Habitable Zone (Liquid Water Possible)",
            "starType": "M-Dwarf (Red Dwarf)",
            "distance": 582
        }
    },
    "kepler_90i": {
        "id": "kepler_90i",
        "name": "Kepler-90i (Multi-Planet System)",
        "type": "Confirmed Exoplanet",
        "description": "A super-Earth orbiting Kepler-90, the first star known to host 8 planets (like our solar system). Exhibits overlapping transit signatures.",
        "metadata": {
            "status": "Transit Detected (Multi-planet)",
            "confidence": 0.95,
            "period": 14.45,
            "depth": 0.15,
            "duration": 2.80,
            "radius": 1.32,
            "habitability": "Too Hot (Inferno Planet)",
            "starType": "G-type Star (Sun-like)",
            "distance": 2840
        }
    },
    "eclipsing_binary": {
        "id": "eclipsing_binary",
        "name": "Eclipsing Binary Star (False Positive)",
        "type": "False Positive",
        "description": "Two stars orbiting each other. The deep V-shaped dips look like transits but are actually another star blocking light. Easily misidentified without high-res models.",
        "metadata": {
            "status": "False Positive (Binary System)",
            "confidence": 0.02,
            "period": 8.52,
            "depth": 6.20,
            "duration": 5.12,
            "radius": 9.84,
            "habitability": "Not Applicable (Stellar Companion)",
            "starType": "Binary (K-dwarf + M-dwarf)",
            "distance": 920
        }
    },
    "stellar_activity": {
        "id": "stellar_activity",
        "name": "Active Starspots (False Positive)",
        "type": "False Positive",
        "description": "Stellar rotation carries large, dark starspots across the stellar disk, creating smooth sinusoidal variability that mimics transits.",
        "metadata": {
            "status": "False Positive (Stellar Noise)",
            "confidence": 0.05,
            "period": 5.20,
            "depth": 0.40,
            "duration": 24.0,
            "radius": 0.0,
            "habitability": "Not Applicable (Starspots)",
            "starType": "Active Solar-type G-star",
            "distance": 430
        }
    },
    "undiscovered_noisy": {
        "id": "undiscovered_noisy",
        "name": "TOI-2026 (Unexplored Noisy Star)",
        "type": "Candidate Exoplanet",
        "description": "A high-priority target with severe instrumental noise and high stellar activity. Denoising exposes a potential habitable super-Earth.",
        "metadata": {
            "status": "Transit Detected (Candidate)",
            "confidence": 0.89,
            "period": 19.82,
            "depth": 0.12,
            "duration": 4.10,
            "radius": 1.64,
            "habitability": "Habitable Zone (Super-Earth)",
            "starType": "K-dwarf (Orange Dwarf)",
            "distance": 145
        }
    }
}


@router.get("/scenarios", response_model=List[ScenarioInfo])
async def list_scenarios():
    """Get all available exoplanet scenarios"""
    return [ScenarioInfo(**s) for s in SCENARIOS.values()]


@router.get("/scenarios/{scenario_id}", response_model=ScenarioInfo)
async def get_scenario(scenario_id: str):
    """Get specific scenario by ID"""
    if scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return ScenarioInfo(**SCENARIOS[scenario_id])


@router.post("/scenarios/{scenario_id}/simulate")
async def simulate_scenario(scenario_id: str, noise_scale: float = 1.0):
    """Generate simulated light curve for a scenario"""
    if scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    # Import the simulator logic
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from utils.simulator import generate_light_curve
    
    try:
        data = generate_light_curve(scenario_id, noise_scale)
        return {
            "scenario_id": scenario_id,
            "noise_scale": noise_scale,
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))