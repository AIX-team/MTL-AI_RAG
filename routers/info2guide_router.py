from fastapi import APIRouter, HTTPException
from models.info2guide_model import PlaceSelectRequest, TravelPlan
from services.info2guide_service import TravelPlannerService
from typing import List

router = APIRouter()
travel_service = TravelPlannerService()

@router.post("/generate-plans", response_model=List[TravelPlan])
async def generate_plans(request: PlaceSelectRequest):
    try:
        print(f"Received request: {request}")  # 디버깅을 위한 로그 추가
        plans = await travel_service.generate_travel_plans(
            places=request.places,
            days=request.travel_days,
            plan_type=request.travel_taste
        )
        return plans
    except Exception as e:
        print(f"Error in generate_plans: {str(e)}")  # 디버깅을 위한 로그 추가
        raise HTTPException(status_code=500, detail=str(e))
