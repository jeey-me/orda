from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# 🔹 산업 정보 (분석 결과에 포함될 항목)
class IndustryInfo(BaseModel):
    name: str
    description: str
    score: float

# 🔹 과거 이슈 정보 (분석 결과에 포함될 항목)
class PastIssueInfo(BaseModel):
    title: str
    content: str
    score: float

# 🔹 분석 요청 (news_content 기반 분석)
class AnalysisRequest(BaseModel):
    news_content: str
    title: str
    content: str
    issue_id: Optional[int] = None
    source: str = "manual"

# 🔹 분석 응답 (산업 + 과거이슈)
class AnalysisResponse(BaseModel):
    related_industries: List[IndustryInfo]
    similar_past_issues: List[PastIssueInfo]

# 🔹 헬스 체크 응답
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]
    version: str

# 🔹 모의투자 시뮬레이션 요청
class SimulationRequest(BaseModel):
    scenario_id: str
    investment_amount: int
    investment_period: int  # 단위: months
    selected_stocks: List[Dict[str, Any]]

# 🔹 모의투자 시뮬레이션 응답
class SimulationResponse(BaseModel):
    scenario_info: Dict[str, Any]
    simulation_results: Dict[str, Any]
    market_comparison: Dict[str, Any]
    stock_analysis: List[Dict[str, Any]]
    learning_points: List[str]
    simulation_metadata: Dict[str, Any]