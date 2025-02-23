from fastapi import APIRouter, HTTPException
from models.info2guide_model import PlaceSelectRequest, GuideBookResponse, CourseList, CoursePlaceResponse
from services.info2guide_service import TravelPlannerService
from typing import List
import uuid

router = APIRouter()
travel_service = TravelPlannerService()

@router.post("/generate-plans", response_model=GuideBookResponse)
async def generate_plans(request: PlaceSelectRequest):
    try:
        # 여행 계획 생성
        plan = await travel_service.generate_travel_plans(
            request.places,
            request.travel_days,
            request.travelTaste
        )
        
        # 코스 목록 생성
        courses = []
        for day_num, day_plan in enumerate(plan.daily_plans, 1):
            course_places = []
            for place_num, place in enumerate(day_plan.places, 1):
                course_place = CoursePlaceResponse(
                    num=place_num,
                    id=place.id,
                    name=place.title,
                    type=place.type,
                    description=place.description,
                    image=place.image,
                    address=place.address,
                    hours=place.open_hours,
                    intro=place.intro,
                    latitude=str(place.latitude),
                    longitude=str(place.longitude)
                )
                course_places.append(course_place)
            
            course = CourseList(
                courseId=str(uuid.uuid4()),
                courseNum=day_num,
                coursePlaces=course_places
            )
            courses.append(course)
        
        # 최종 응답 생성
        response = GuideBookResponse(
            success="success",
            message="success",
            guideBookTitle=f"가이드북 {len(courses)}일차",
            travelInfoTitle="여행 정보",
            travelInfoId=str(uuid.uuid4()),
            courseCnt=len(courses),
            courses=courses
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
