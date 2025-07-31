#!/usr/bin/env python3
"""
Stock Market Relevance Filter
70개 이슈 중 주식시장과 가장 밀접한 5개 이슈를 AI로 선별
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 환경변수 로드
load_dotenv(override=True)

class StockMarketFilter:
    """주식시장 관련성 기반 이슈 필터링 클래스"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        필터 초기화
        
        Args:
            model: 사용할 OpenAI 모델
            temperature: 모델 temperature 설정
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.filter_prompt = self._create_filter_prompt()
        self.detailed_analysis_prompt = self._create_detailed_analysis_prompt()
        
        print(f"🤖 StockMarketFilter 초기화 완료 (모델: {model})")

    def _create_filter_prompt(self) -> ChatPromptTemplate:
        """주식시장 관련성 필터링용 프롬프트 생성"""
        return ChatPromptTemplate.from_messages([
            ("system", """너는 한국 주식시장 전문 애널리스트야. 
주어진 뉴스 이슈들을 분석하여 주식시장에 가장 큰 영향을 미칠 것으로 예상되는 이슈들을 선별해야 해.

📊 평가 기준 (각 1-10점):
1. **직접적 기업 영향**: 특정 기업이나 산업의 실적에 직접적인 영향을 미치는가?
2. **정책적 영향**: 금리, 세금, 규제 변화 등 시장 전반에 영향을 미치는 정책인가?
3. **시장 심리**: 투자자 신뢰도, 리스크 인식, 투자 심리에 미치는 영향은?
4. **거시경제**: GDP, 인플레이션, 환율 등 거시경제 지표에 미치는 영향은?
5. **산업 트렌드**: 새로운 기술이나 소비 패턴 변화로 인한 산업 영향은?

💡 우선순위:
- 단기적 주가 변동을 일으킬 가능성이 높은 이슈
- 특정 업종이나 테마주에 영향을 미치는 이슈
- 외국인 투자나 기관 투자에 영향을 미치는 이슈
- 정부 정책이나 규제 변화 관련 이슈"""),
            
            ("human", """다음 {total_count}개의 뉴스 이슈 중에서 주식시장에 가장 큰 영향을 미칠 것으로 예상되는 상위 5개를 선별해주세요.

📰 이슈 목록:
{issues_list}

각 이슈에 대해 다음 기준으로 평가하고, 상위 5개를 선택해주세요:
- 종합 점수 (1-10점)
- 주된 영향 분야 (예: 금융주, 바이오주, 반도체 등)
- 예상 영향 방향 (긍정적/부정적/중립적)
- 영향 시기 (즉시/단기/중기)

출력 형식 (JSON):
{{
  "analysis_summary": {{
    "total_analyzed": {total_count},
    "selected_count": 5,
    "analysis_timestamp": "현재 시간",
    "filter_confidence": "분석 신뢰도 (1-10점)"
  }},
  "selected_issues": [
    {{
      "rank": 1,
      "이슈번호": 원본_이슈번호,
      "카테고리": "원본_카테고리",
      "제목": "원본_제목",
      "종합점수": 9.2,
      "직접기업영향": 9,
      "정책영향": 8,
      "시장심리": 9,
      "거시경제": 7,
      "산업트렌드": 8,
      "주된영향분야": ["금융주", "은행주"],
      "예상영향방향": "긍정적",
      "영향시기": "즉시",
      "선별이유": "구체적인 선별 이유 설명",
      "예상시장반응": "예상되는 시장 반응 설명"
    }},
    ...
  ],
  "filtering_notes": "전반적인 선별 과정에서의 특이사항이나 고려사항"
}}""")
        ])

    def _create_detailed_analysis_prompt(self) -> ChatPromptTemplate:
        """선별된 이슈들의 상세 분석용 프롬프트"""
        return ChatPromptTemplate.from_messages([
            ("system", """너는 한국 주식시장 전문 애널리스트야. 
선별된 주요 이슈들에 대해 더 자세한 시장 영향 분석을 제공해야 해."""),
            
            ("human", """다음 5개 이슈에 대해 상세한 주식시장 영향 분석을 제공해주세요:

{selected_issues}

각 이슈별로 다음 내용을 포함해서 분석해주세요:
1. 핵심 영향 요인
2. 영향받을 구체적인 종목/섹터
3. 과거 유사 사례와의 비교
4. 투자 전략 제안
5. 리스크 요인

출력 형식 (JSON):
{{
  "detailed_analysis": [
    {{
      "rank": 1,
      "제목": "이슈 제목",
      "핵심영향요인": ["요인1", "요인2"],
      "영향섹터": [
        {{"섹터명": "금융", "영향도": "높음", "방향": "긍정적"}},
        {{"섹터명": "부동산", "영향도": "중간", "방향": "부정적"}}
      ],
      "관련종목예시": ["삼성전자", "SK하이닉스"],
      "과거유사사례": "2020년 코로나19 금리 인하 시 금융주 급등",
      "투자전략": "단기적으로 금융주 매수 포지션 고려",
      "리스크요인": ["정책 변화 가능성", "외부 경제 변수"],
      "신뢰도": 8.5
    }}
  ],
  "market_outlook": {{
    "overall_sentiment": "긍정적/부정적/중립적",
    "key_themes": ["주요 테마1", "주요 테마2"],
    "attention_sectors": ["주목할 섹터들"],
    "risk_factors": ["전반적 리스크 요인들"]
  }}
}}""")
        ])

    def filter_issues_by_stock_relevance(self, issues_data: List[Dict], target_count: int = 5) -> Dict:
        """
        주식시장 관련성을 기준으로 이슈 필터링
        
        Args:
            issues_data: 크롤링된 전체 이슈 리스트
            target_count: 선별할 이슈 수
            
        Returns:
            필터링 결과 딕셔너리
        """
        if not issues_data:
            raise ValueError("필터링할 이슈 데이터가 없습니다.")
        
        print(f"🔍 주식시장 관련성 필터링 시작")
        print(f"📊 분석 대상: {len(issues_data)}개 이슈")
        print(f"🎯 선별 목표: {target_count}개 이슈")
        
        try:
            # 1. 이슈 목록을 텍스트로 변환
            issues_text = self._format_issues_for_analysis(issues_data)
            
            # 2. AI 필터링 실행
            print("🤖 AI 분석 실행 중...")
            filtering_result = self._execute_filtering(issues_text, len(issues_data))
            
            # 3. 결과 검증 및 보완
            validated_result = self._validate_filtering_result(filtering_result, issues_data)
            
            # 4. 상세 분석 추가
            print("📈 상세 시장 영향 분석 중...")
            detailed_analysis = self._generate_detailed_analysis(validated_result["selected_issues"])
            
            # 5. 최종 결과 구성
            final_result = {
                "filter_metadata": {
                    "filtered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_analyzed": len(issues_data),
                    "selected_count": len(validated_result["selected_issues"]),
                    "filter_model": "gpt-4o",
                    "filter_confidence": validated_result.get("analysis_summary", {}).get("filter_confidence", "N/A")
                },
                "selected_issues": validated_result["selected_issues"],
                "detailed_analysis": detailed_analysis.get("detailed_analysis", []),
                "market_outlook": detailed_analysis.get("market_outlook", {}),
                "filtering_notes": validated_result.get("filtering_notes", ""),
                "original_issues_count": len(issues_data)
            }
            
            print(f"✅ 필터링 완료: {len(final_result['selected_issues'])}개 이슈 선별")
            self._print_filtering_summary(final_result)
            
            return final_result
            
        except Exception as e:
            print(f"❌ 필터링 실패: {e}")
            raise

    def _format_issues_for_analysis(self, issues_data: List[Dict]) -> str:
        """이슈 데이터를 AI 분석용 텍스트로 변환"""
        formatted_issues = []
        
        for issue in issues_data:
            issue_text = f"""
이슈번호: {issue.get('이슈번호', 'N/A')}
카테고리: {issue.get('카테고리', 'N/A')}
제목: {issue.get('제목', 'N/A')}
내용: {issue.get('내용', 'N/A')[:200]}...
"""
            formatted_issues.append(issue_text.strip())
        
        return "\n" + "="*50 + "\n".join(formatted_issues)

    def _execute_filtering(self, issues_text: str, total_count: int) -> Dict:
        """AI 필터링 실행"""
        try:
            parser = JsonOutputParser()
            chain = self.filter_prompt | self.llm | parser
            
            result = chain.invoke({
                "issues_list": issues_text,
                "total_count": total_count
            })
            
            return result
            
        except Exception as e:
            print(f"❌ AI 필터링 실행 실패: {e}")
            raise

    def _validate_filtering_result(self, filtering_result: Dict, original_issues: List[Dict]) -> Dict:
        """필터링 결과 검증 및 보완"""
        try:
            selected_issues = filtering_result.get("selected_issues", [])
            
            # 선별된 이슈에 원본 데이터 보완
            for selected in selected_issues:
                issue_num = selected.get("이슈번호")
                original_issue = next((issue for issue in original_issues if issue.get("이슈번호") == issue_num), None)
                
                if original_issue:
                    # 원본 데이터로 보완
                    selected["원본내용"] = original_issue.get("내용", "")
                    selected["추출시간"] = original_issue.get("추출시간", "")
                    selected["고유ID"] = original_issue.get("고유ID", "")
                else:
                    print(f"⚠️ 이슈번호 {issue_num}에 대한 원본 데이터를 찾을 수 없습니다.")
            
            # 순위 검증 및 정렬
            selected_issues.sort(key=lambda x: x.get("rank", 999))
            
            # 필수 필드 검증
            for i, issue in enumerate(selected_issues, 1):
                if "rank" not in issue:
                    issue["rank"] = i
                if "종합점수" not in issue:
                    issue["종합점수"] = 7.0  # 기본값
            
            filtering_result["selected_issues"] = selected_issues
            return filtering_result
            
        except Exception as e:
            print(f"⚠️ 필터링 결과 검증 중 오류: {e}")
            return filtering_result

    def _generate_detailed_analysis(self, selected_issues: List[Dict]) -> Dict:
        """선별된 이슈들에 대한 상세 분석 생성"""
        try:
            # 상세 분석용 데이터 포맷팅
            issues_summary = []
            for issue in selected_issues:
                summary = f"""
순위 {issue.get('rank', 'N/A')}: {issue.get('제목', 'N/A')}
카테고리: {issue.get('카테고리', 'N/A')}
종합점수: {issue.get('종합점수', 'N/A')}
선별이유: {issue.get('선별이유', 'N/A')}
"""
                issues_summary.append(summary.strip())
            
            issues_text = "\n" + "="*30 + "\n".join(issues_summary)
            
            parser = JsonOutputParser()
            chain = self.detailed_analysis_prompt | self.llm | parser
            
            result = chain.invoke({
                "selected_issues": issues_text
            })
            
            return result
            
        except Exception as e:
            print(f"⚠️ 상세 분석 생성 실패: {e}")
            return {"detailed_analysis": [], "market_outlook": {}}

    def _print_filtering_summary(self, result: Dict):
        """필터링 결과 요약 출력"""
        print(f"\n{'='*80}")
        print(f"📊 주식시장 관련성 필터링 결과")
        print(f"{'='*80}")
        
        metadata = result["filter_metadata"]
        print(f"🕐 필터링 시간: {metadata['filtered_at']}")
        print(f"📈 분석 모델: {metadata['filter_model']}")
        print(f"🎯 선별 비율: {metadata['selected_count']}/{metadata['total_analyzed']} ({metadata['selected_count']/metadata['total_analyzed']*100:.1f}%)")
        print(f"💯 필터 신뢰도: {metadata['filter_confidence']}")
        
        print(f"\n🏆 선별된 상위 {len(result['selected_issues'])}개 이슈:")
        for issue in result["selected_issues"]:
            print(f"   {issue['rank']}. [{issue['카테고리']}] {issue['제목'][:50]}...")
            print(f"      💰 종합점수: {issue['종합점수']}/10")
            print(f"      📊 주된영향: {', '.join(issue.get('주된영향분야', ['N/A']))}")
            print(f"      ⏰ 영향시기: {issue.get('영향시기', 'N/A')}")
            print()
        
        # 시장 전망 요약
        outlook = result.get("market_outlook", {})
        if outlook:
            print(f"🔮 전반적 시장 전망:")
            print(f"   • 시장 심리: {outlook.get('overall_sentiment', 'N/A')}")
            print(f"   • 주요 테마: {', '.join(outlook.get('key_themes', []))}")
            print(f"   • 주목 섹터: {', '.join(outlook.get('attention_sectors', []))}")

    def save_filtered_results(self, results: Dict, data_dir: str = "data2") -> str:
        """필터링 결과를 JSON 파일로 저장"""
        try:
            data_path = Path(data_dir)
            data_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            filename = f"{timestamp}_StockFiltered_5issues.json"
            filepath = data_path / filename
            
            save_data = {
                **results,
                "file_info": {
                    "filename": filename,
                    "created_at": datetime.now().isoformat(),
                    "filter_version": "StockMarketFilter_v1.0"
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 필터링 결과 저장: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
            raise

    def load_latest_filtered_results(self, data_dir: str = "data2") -> Optional[Dict]:
        """최신 필터링 결과 로드"""
        try:
            data_path = Path(data_dir)
            json_files = list(data_path.glob("*_StockFiltered_*issues.json"))
            
            if not json_files:
                print("📂 저장된 필터링 결과가 없습니다.")
                return None
            
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ 최신 필터링 결과 로드: {latest_file.name}")
            return data
            
        except Exception as e:
            print(f"❌ 필터링 결과 로드 실패: {e}")
            return None

# 편의 함수들
def filter_issues_for_stock_market(issues_data: List[Dict], target_count: int = 5) -> Dict:
    """주식시장 관련성 기준 이슈 필터링 편의 함수"""
    filter_system = StockMarketFilter()
    return filter_system.filter_issues_by_stock_relevance(issues_data, target_count)

def load_and_filter_latest_crawled_data(data_dir: str = "data2") -> Optional[Dict]:
    """최신 크롤링 데이터를 로드하여 필터링"""
    from crawling_bigkinds import BigKindsCrawler  # 변경된 부분
    
    # 최신 크롤링 데이터 로드
    crawled_data = load_latest_multi_data()
    if not crawled_data or not crawled_data.get("all_issues"):
        print("❌ 크롤링 데이터를 찾을 수 없습니다.")
        return None
    
    # 필터링 실행
    filter_system = StockMarketFilter()
    filtered_result = filter_system.filter_issues_by_stock_relevance(crawled_data["all_issues"])
    
    # 결과 저장
    saved_file = filter_system.save_filtered_results(filtered_result, data_dir)
    filtered_result["saved_file"] = saved_file
    
    return filtered_result

# 메인 실행
if __name__ == "__main__":
    print("🤖 Stock Market Relevance Filter")
    
    try:
        # 최신 크롤링 데이터 로드 및 필터링
        print("📥 최신 크롤링 데이터 로드 중...")
        result = load_and_filter_latest_crawled_data()
        
        if result:
            print(f"\n🎯 필터링 완료!")
            print(f"   • 저장 파일: {result.get('saved_file', 'N/A')}")
            print(f"   • 선별된 이슈 수: {len(result.get('selected_issues', []))}개")
            
            # 선별된 이슈 간단 미리보기
            selected = result.get("selected_issues", [])
            if selected:
                print(f"\n🏆 TOP 3 이슈:")
                for issue in selected[:3]:
                    print(f"   {issue['rank']}. {issue['제목'][:50]}... (점수: {issue['종합점수']})")
        else:
            print("❌ 필터링할 데이터가 없습니다.")
            
    except Exception as e:
        print(f"❌ 필터링 실행 실패: {e}")
        import traceback
        traceback.print_exc()