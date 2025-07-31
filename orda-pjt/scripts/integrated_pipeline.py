#!/usr/bin/env python3
"""
Complete Integrated Pipeline for BigKinds News Analysis with RAG
전체 프로세스: 크롤링 → AI 필터링 → 실제 RAG 분석 → API 준비
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import traceback
import pandas as pd

# 환경 변수 및 AI 모델 설정
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 모듈 임포트
try:
    from crawling_bigkinds import BigKindsCrawler
    from stock_market_filter import StockMarketFilter
except ImportError as e:
    print(f"⚠️ 모듈 임포트 실패: {e}")
    print("필요한 파일들이 같은 디렉토리에 있는지 확인해주세요:")
    print("- crawling_bigkinds.py")
    print("- stock_market_filter.py")

class RealRAGAnalysisExecutor:
    """실제 RAG 분석을 실행하고 결과를 파싱하는 클래스"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        load_dotenv(override=True)
        
        # 환경 설정
        self.EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lastproject")
        
        # LLM 초기화 (GPT-4o-mini 사용)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embedding = OpenAIEmbeddings(model=self.EMBEDDING_MODEL)
        
        # 벡터 스토어 초기화
        self.industry_store = PineconeVectorStore(
            index_name=self.INDEX_NAME,
            embedding=self.embedding,
            namespace="industry"
        )
        
        self.past_issue_store = PineconeVectorStore(
            index_name=self.INDEX_NAME,
            embedding=self.embedding,
            namespace="past_issue"
        )
        

        # 산업 DB 로딩 (상대 경로 수정)
        try:
            # scripts 폴더에서 실행되므로 ../data/ 경로 사용
            self.industry_df = pd.read_csv("../data/산업DB.v.0.3.csv")
            self.industry_dict = dict(zip(self.industry_df["KRX 업종명"], self.industry_df["상세내용"]))
            self.valid_krx_names = list(self.industry_df["KRX 업종명"].unique())
            print(f"✅ 산업 DB 로드: {len(self.valid_krx_names)}개 업종")
        except Exception as e:
            print(f"⚠️ 산업 DB 로드 실패: {e}")
            self.industry_dict = {}
            self.valid_krx_names = []
        
        # 과거 이슈 DB 로딩 (상대 경로 수정)
        try:
            # scripts 폴더에서 실행되므로 ../data/ 경로 사용
            self.past_df = pd.read_csv("../data/Past_news.csv")
            self.issue_dict = dict(zip(self.past_df["Issue_name"], self.past_df["Contents"] + "\n\n상세: " + self.past_df["Contentes(Spec)"]))
            self.valid_issue_names = list(self.past_df["Issue_name"].unique())
            print(f"✅ 과거 이슈 DB 로드: {len(self.valid_issue_names)}개 이슈")
        except Exception as e:
            print(f"⚠️ 과거 이슈 DB 로드 실패: {e}")
            self.issue_dict = {}
            self.valid_issue_names = []
        
        print("✅ 실제 RAG 분석 시스템 초기화 완료 (GPT-4o-mini)")
    
    def analyze_industry_for_issue(self, issue: Dict) -> List[Dict]:
        """특정 이슈에 대한 관련 산업 분석 (실제 RAG)"""
        try:
            query = f"{issue.get('제목', '')}\n{issue.get('원본내용', issue.get('내용', ''))}"
            print(f"🏭 산업 분석 중: {issue.get('제목', 'N/A')[:30]}...")
            
            # Step 1: 벡터 검색으로 후보 추출
            results = self.industry_store.similarity_search_with_score(query, k=10)
            
            vector_candidates = []
            for doc, score in results:
                content = doc.page_content.replace('\ufeff', '').replace('﻿', '')
                
                if "KRX 업종명:" in content:
                    lines = content.split("\n")
                    for line in lines:
                        if "KRX 업종명:" in line:
                            industry_name = line.replace("KRX 업종명:", "").strip()
                            if industry_name in self.industry_dict:
                                # 중복 체크
                                if not any(c["name"] == industry_name for c in vector_candidates):
                                    similarity_percentage = round((1 - score) * 100, 1)
                                    
                                    content_parts = content.split("상세내용:")
                                    industry_detail = content_parts[1].strip() if len(content_parts) > 1 else self.industry_dict[industry_name]
                                    
                                    vector_candidates.append({
                                        "name": industry_name,
                                        "similarity": similarity_percentage,
                                        "description": industry_detail
                                    })
                            break
            
            # Step 2: AI Agent로 관련 산업 후보 추출
            ai_candidates = self._extract_candidate_industries(query, self.valid_krx_names, top_k=10)
            
            # Step 3: 결과 결합 및 검증
            final_candidates = self._combine_and_validate_industry_results(
                query, vector_candidates, ai_candidates, self.industry_dict
            )
            
            print(f"   ✅ 산업 분석 완료: 상위 {min(3, len(final_candidates))}개 선택")
            return final_candidates[:3]  # 상위 3개 반환
            
        except Exception as e:
            print(f"❌ 산업 분석 실패: {e}")
            return []
    
    def analyze_past_issues_for_issue(self, issue: Dict) -> List[Dict]:
        """특정 이슈에 대한 관련 과거 이슈 분석 (실제 RAG)"""
        try:
            query = f"{issue.get('제목', '')}\n{issue.get('원본내용', issue.get('내용', ''))}"
            print(f"📚 과거 이슈 분석 중: {issue.get('제목', 'N/A')[:30]}...")
            
            # Step 1: 벡터 검색으로 후보 추출
            results = self.past_issue_store.similarity_search_with_score(query, k=10)
            
            vector_candidates = []
            for doc, score in results:
                content = doc.page_content.replace('\ufeff', '').replace('﻿', '')
                
                if "Issue_name:" in content:
                    lines = content.split("\n")
                    for line in lines:
                        if "Issue_name:" in line:
                            issue_name = line.replace("Issue_name:", "").strip()
                            if issue_name in self.issue_dict:
                                # 중복 체크
                                if not any(c["name"] == issue_name for c in vector_candidates):
                                    similarity_percentage = round((1 - score) * 100, 1)
                                    
                                    content_parts = content.split("Contents:")
                                    issue_detail = content_parts[1].strip() if len(content_parts) > 1 else self.issue_dict[issue_name]
                                    
                                    # 기간 정보 추출
                                    period = "N/A"
                                    for line in lines:
                                        if "Start_date:" in line and "Fin_date:" in line:
                                            start = line.split("Start_date:")[1].split("Fin_date:")[0].strip()
                                            end = line.split("Fin_date:")[1].strip()
                                            period = f"{start} ~ {end}"
                                            break
                                    
                                    vector_candidates.append({
                                        "name": issue_name,
                                        "similarity": similarity_percentage,
                                        "description": issue_detail,
                                        "period": period
                                    })
                            break
            
            # Step 2: AI Agent로 관련 과거 이슈 후보 추출
            ai_candidates = self._extract_candidate_past_issues(query, self.valid_issue_names, top_k=10)
            
            # Step 3: 결과 결합 및 검증
            final_candidates = self._combine_and_validate_past_issue_results(
                query, vector_candidates, ai_candidates, self.issue_dict
            )
            
            print(f"   ✅ 과거 이슈 분석 완료: 상위 {min(3, len(final_candidates))}개 선택")
            return final_candidates[:3]  # 상위 3개 반환
            
        except Exception as e:
            print(f"❌ 과거 이슈 분석 실패: {e}")
            return []
    
    def _extract_candidate_industries(self, news_content: str, industry_list: List[str], top_k: int = 10) -> List[Dict]:
        """AI Agent가 뉴스 내용을 보고 관련 가능성이 높은 산업들을 추출"""
        if not industry_list:
            return []
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 뉴스와 산업의 관련성을 판단하는 전문 애널리스트야.
주어진 뉴스 내용을 분석하고, 제공된 KRX 업종 리스트에서 관련 가능성이 높은 산업들을 선별해야 해.

관련성 판단 기준:
1. 직접적 영향: 뉴스가 해당 산업에 직접적인 영향을 미치는가?
2. 공급망 관계: 뉴스 관련 기업/산업과 공급망 관계가 있는가?
3. 시장 동향: 뉴스가 해당 산업의 시장 동향에 영향을 미치는가?
4. 정책/규제: 뉴스가 해당 산업 관련 정책이나 규제와 연관되는가?"""),
            ("human", """
[뉴스 내용]
{news}

[KRX 업종 리스트]  
{industries}

위 뉴스와 관련 가능성이 높은 산업을 {top_k}개 선별해주세요.
각 산업에 대해 관련성 점수(1-10점)와 간단한 이유를 제시해주세요.

출력 형식 (JSON):
{{
  "candidates": [
    {{"industry": "산업명", "score": 점수, "reason": "관련성 이유"}},
    ...
  ]
}}""")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "news": news_content,
                "industries": ", ".join(industry_list[:50]),  # 너무 많으면 제한
                "top_k": top_k
            })
            return result.get("candidates", [])
        except Exception as e:
            print(f"❌ AI 산업 후보 추출 실패: {e}")
            return []
    
    def _extract_candidate_past_issues(self, news_content: str, issue_list: List[str], top_k: int = 10) -> List[Dict]:
        """AI Agent가 뉴스 내용을 보고 관련 가능성이 높은 과거 이슈들을 추출"""
        if not issue_list:
            return []
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 현재 뉴스와 과거 이슈의 관련성을 판단하는 전문 애널리스트야.
주어진 현재 뉴스 내용을 분석하고, 제공된 과거 이슈 리스트에서 관련 가능성이 높은 이슈들을 선별해야 해.

관련성 판단 기준:
1. 유사한 시장 상황: 과거 이슈와 현재 상황이 유사한 시장 환경인가?
2. 동일한 산업/기업 영향: 같은 산업이나 유사한 기업들에 영향을 미치는가?
3. 정책/경제적 유사성: 정책 변화나 경제적 요인이 유사한가?
4. 투자자 심리: 투자자들의 반응이나 시장 심리가 비슷한가?"""),
            ("human", """
[현재 뉴스 내용]
{news}

[과거 이슈 리스트]
{issues}

위 현재 뉴스와 관련 가능성이 높은 과거 이슈를 {top_k}개 선별해주세요.
각 과거 이슈에 대해 관련성 점수(1-10점)와 간단한 이유를 제시해주세요.

출력 형식 (JSON):
{{
  "candidates": [
    {{"issue": "이슈명", "score": 점수, "reason": "관련성 이유"}},
    ...
  ]
}}""")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "news": news_content,
                "issues": ", ".join(issue_list[:50]),  # 너무 많으면 제한
                "top_k": top_k
            })
            return result.get("candidates", [])
        except Exception as e:
            print(f"❌ AI 과거 이슈 후보 추출 실패: {e}")
            return []
    
    def _combine_and_validate_industry_results(self, news_content: str, vector_candidates: List[Dict], 
                                             ai_candidates: List[Dict], industry_dict: Dict) -> List[Dict]:
        """벡터 검색 결과와 AI 후보를 결합하여 최종 관련 산업 도출"""
        all_candidates = {}
        
        # 벡터 검색 결과 추가
        for candidate in vector_candidates:
            name = candidate["name"]
            all_candidates[name] = {
                "name": name,
                "vector_similarity": candidate["similarity"],
                "ai_score": 0,
                "ai_reason": "",
                "description": candidate["description"]
            }
        
        # AI 후보 추가/업데이트
        for candidate in ai_candidates:
            name = candidate["industry"]
            if name in all_candidates:
                all_candidates[name]["ai_score"] = candidate["score"]
                all_candidates[name]["ai_reason"] = candidate["reason"]
            elif name in industry_dict:  # 유효한 산업명인 경우만
                all_candidates[name] = {
                    "name": name,
                    "vector_similarity": 0,
                    "ai_score": candidate["score"],
                    "ai_reason": candidate["reason"],
                    "description": industry_dict[name]
                }
        
        # 종합 점수 계산 (벡터 유사도 + AI 점수)
        for candidate in all_candidates.values():
            # 벡터 유사도를 10점 만점으로 정규화
            normalized_vector = candidate["vector_similarity"] / 10
            # AI 점수는 이미 10점 만점
            ai_score = candidate["ai_score"]
            
            # 가중평균 (AI 점수에 더 높은 가중치)
            candidate["final_score"] = round((normalized_vector * 0.3 + ai_score * 0.7), 1)
        
        # 최종 점수로 정렬
        sorted_candidates = sorted(all_candidates.values(), 
                                  key=lambda x: x["final_score"], 
                                  reverse=True)
        
        return sorted_candidates
    
    def _combine_and_validate_past_issue_results(self, news_content: str, vector_candidates: List[Dict], 
                                               ai_candidates: List[Dict], issue_dict: Dict) -> List[Dict]:
        """벡터 검색 결과와 AI 후보를 결합하여 최종 관련 과거 이슈 도출"""
        all_candidates = {}
        
        # 벡터 검색 결과 추가
        for candidate in vector_candidates:
            name = candidate["name"]
            all_candidates[name] = {
                "name": name,
                "vector_similarity": candidate["similarity"],
                "ai_score": 0,
                "ai_reason": "",
                "description": candidate["description"],
                "period": candidate.get("period", "N/A")
            }
        
        # AI 후보 추가/업데이트
        for candidate in ai_candidates:
            name = candidate["issue"]
            if name in all_candidates:
                all_candidates[name]["ai_score"] = candidate["score"]
                all_candidates[name]["ai_reason"] = candidate["reason"]
            elif name in issue_dict:  # 유효한 이슈명인 경우만
                all_candidates[name] = {
                    "name": name,
                    "vector_similarity": 0,
                    "ai_score": candidate["score"],
                    "ai_reason": candidate["reason"],
                    "description": issue_dict[name],
                    "period": "N/A"
                }
        
        # 종합 점수 계산
        for candidate in all_candidates.values():
            normalized_vector = candidate["vector_similarity"] / 10
            ai_score = candidate["ai_score"]
            candidate["final_score"] = round((normalized_vector * 0.3 + ai_score * 0.7), 1)
        
        # 최종 점수로 정렬
        sorted_candidates = sorted(all_candidates.values(), 
                                  key=lambda x: x["final_score"], 
                                  reverse=True)
        
        return sorted_candidates

class IntegratedNewsPipeline:
    """통합 뉴스 분석 파이프라인 (실제 RAG 분석 포함)"""
    
    def __init__(self, data_dir: str = "data2", headless: bool = True):
        """
        파이프라인 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
            headless: 크롤링 시 헤드리스 모드 사용 여부
        """
        project_root = Path(__file__).parent.parent  # scripts의 상위 폴더
        self.data_dir = project_root / data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.headless = headless
        
        # 컴포넌트 초기화
        self.crawler = None
        self.filter = None
        self.rag_executor = RealRAGAnalysisExecutor(str(self.data_dir))  # 실제 RAG 실행기 사용
        
        # 실행 결과 저장
        self.pipeline_results = {
            "pipeline_id": None,
            "started_at": None,
            "completed_at": None,
            "execution_time": None,
            "steps_completed": [],
            "final_status": None,
            "crawling_result": None,
            "filtering_result": None,
            "rag_analysis_result": None,
            "api_ready_data": None,
            "errors": []
        }

    def run_full_pipeline(self, 
                         issues_per_category: int = 10,
                         target_filtered_count: int = 5,
                         force_new_crawling: bool = False) -> Dict:
        """
        전체 파이프라인 실행 (실제 RAG 분석 포함)
        
        Args:
            issues_per_category: 카테고리별 크롤링할 이슈 수
            target_filtered_count: 필터링할 최종 이슈 수
            force_new_crawling: 강제 새 크롤링 여부
            
        Returns:
            파이프라인 실행 결과
        """
        # 파이프라인 시작
        pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipeline_results["pipeline_id"] = pipeline_id
        self.pipeline_results["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"🚀 통합 뉴스 분석 파이프라인 시작 (실제 RAG + GPT-4o-mini, ID: {pipeline_id})")
        print(f"{'='*80}")
        print(f"📋 파이프라인 설정:")
        print(f"   • 카테고리별 이슈 수: {issues_per_category}개")
        print(f"   • 최종 선별 이슈 수: {target_filtered_count}개")
        print(f"   • 실제 RAG 분석 포함: 예")
        print(f"   • AI 모델: GPT-4o-mini")
        print(f"   • 강제 새 크롤링: {'예' if force_new_crawling else '아니오'}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: 크롤링 (또는 기존 데이터 로드)
            crawling_result = self._execute_crawling_step(
                issues_per_category, force_new_crawling
            )
            
            # Step 2: AI 필터링
            filtering_result = self._execute_filtering_step(
                crawling_result, target_filtered_count
            )
            
            # Step 3: 실제 RAG 분석
            rag_result = self._execute_real_rag_analysis_step(filtering_result)
            
            # Step 4: API용 데이터 준비
            api_data = self._prepare_api_data(rag_result)
            
            # 파이프라인 완료
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            self.pipeline_results.update({
                "completed_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": str(execution_time),
                "final_status": "success",
                "crawling_result": crawling_result,
                "filtering_result": filtering_result,
                "rag_analysis_result": rag_result,
                "api_ready_data": api_data
            })
            
            # 결과 저장
            saved_file = self._save_pipeline_results()
            self.pipeline_results["saved_file"] = saved_file
            
            self._print_pipeline_summary()
            
            return self.pipeline_results
            
        except Exception as e:
            error_msg = f"파이프라인 실행 실패: {e}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            self.pipeline_results.update({
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_status": "failed",
                "errors": [error_msg]
            })
            
            raise

    def _execute_crawling_step(self, issues_per_category: int, force_new: bool) -> Dict:
        """Step 1: 크롤링 실행"""
        print(f"\n{'='*60}")
        print(f"📡 Step 1: 다중 카테고리 크롤링")
        print(f"{'='*60}")
        
        try:
            # 기존 데이터 확인 (force_new가 False인 경우)
            if not force_new:
                existing_data = self._check_recent_crawling_data()
                if existing_data:
                    print(f"♻️ 최근 크롤링 데이터 발견, 재사용합니다.")
                    self.pipeline_results["steps_completed"].append("crawling_reused")
                    return existing_data
            
            # 새 크롤링 실행
            print(f"🕷️ 새로운 크롤링 시작...")
            self.crawler = BigKindsCrawler(
                data_dir=str(self.data_dir),
                headless=self.headless,
                issues_per_category=issues_per_category
            )
            
            crawling_result = self.crawler.crawl_all_categories()
            
            print(f"✅ Step 1 완료: {crawling_result['total_issues']}개 이슈 수집")
            self.pipeline_results["steps_completed"].append("crawling_new")
            
            return crawling_result
            
        except Exception as e:
            error_msg = f"크롤링 단계 실패: {e}"
            self.pipeline_results["errors"].append(error_msg)
            raise Exception(error_msg)

    def _execute_filtering_step(self, crawling_result: Dict, target_count: int) -> Dict:
        """Step 2: AI 필터링 실행"""
        print(f"\n{'='*60}")
        print(f"🤖 Step 2: AI 주식시장 관련성 필터링")
        print(f"{'='*60}")
        
        try:
            if not crawling_result or not crawling_result.get("all_issues"):
                raise ValueError("크롤링 데이터가 없습니다.")
            
            print(f"📊 필터링 대상: {len(crawling_result['all_issues'])}개 이슈")
            print(f"🎯 선별 목표: {target_count}개 이슈")
            
            self.filter = StockMarketFilter()
            filtering_result = self.filter.filter_issues_by_stock_relevance(
                crawling_result["all_issues"], target_count
            )
            
            # 필터링 결과 저장
            saved_filter_file = self.filter.save_filtered_results(
                filtering_result, str(self.data_dir)
            )
            filtering_result["saved_file"] = saved_filter_file
            
            print(f"✅ Step 2 완료: {len(filtering_result['selected_issues'])}개 이슈 선별")
            self.pipeline_results["steps_completed"].append("filtering")
            
            return filtering_result
            
        except Exception as e:
            error_msg = f"필터링 단계 실패: {e}"
            self.pipeline_results["errors"].append(error_msg)
            raise Exception(error_msg)

    def _execute_real_rag_analysis_step(self, filtering_result: Dict) -> Dict:
        """Step 3: 실제 RAG 분석 실행"""
        print(f"\n{'='*60}")
        print(f"🔍 Step 3: 실제 RAG 분석 (산업 + 과거 이슈)")
        print(f"{'='*60}")
        
        try:
            selected_issues = filtering_result.get("selected_issues", [])
            if not selected_issues:
                raise ValueError("필터링된 이슈가 없습니다.")
            
            print(f"📊 실제 RAG 분석 대상: {len(selected_issues)}개 선별 이슈")
            
            # 각 이슈별로 실제 RAG 분석 실행
            enriched_issues = []
            
            for i, issue in enumerate(selected_issues, 1):
                print(f"🔄 이슈 {i}/{len(selected_issues)} 분석 중: {issue.get('제목', 'N/A')[:50]}...")
                
                # 실제 산업 분석
                related_industries = self.rag_executor.analyze_industry_for_issue(issue)
                
                # 실제 과거 이슈 분석
                related_past_issues = self.rag_executor.analyze_past_issues_for_issue(issue)
                
                # RAG 분석 신뢰도 계산
                rag_confidence = self._calculate_rag_confidence(related_industries, related_past_issues)
                
                # 기본 이슈 정보에 실제 RAG 결과 추가
                enriched_issue = issue.copy()
                enriched_issue["관련산업"] = related_industries
                enriched_issue["관련과거이슈"] = related_past_issues
                enriched_issue["RAG분석신뢰도"] = rag_confidence
                
                enriched_issues.append(enriched_issue)
                
                print(f"   ✅ 이슈 {i} RAG 분석 완료: 산업 {len(related_industries)}개, 과거이슈 {len(related_past_issues)}개, 신뢰도 {rag_confidence}")
            
            rag_result = {
                **filtering_result,  # 기존 필터링 결과 유지
                "selected_issues": enriched_issues,  # 실제 RAG 분석 결과로 교체
                "rag_metadata": {
                    "real_rag_analysis": True,
                    "analysis_method": "vector_search + AI_agent + GPT-4o-mini",
                    "rag_completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "enriched_issues_count": len(enriched_issues),
                    "average_confidence": round(sum(issue.get("RAG분석신뢰도", 0) for issue in enriched_issues) / len(enriched_issues), 2) if enriched_issues else 0
                }
            }
            
            # RAG 분석 결과 저장
            saved_rag_file = self._save_rag_results(rag_result)
            rag_result["rag_saved_file"] = saved_rag_file
            
            print(f"✅ Step 3 완료: {len(enriched_issues)}개 이슈 실제 RAG 분석 완료")
            print(f"📊 평균 RAG 신뢰도: {rag_result['rag_metadata']['average_confidence']}")
            self.pipeline_results["steps_completed"].append("real_rag_analysis")
            
            return rag_result
            
        except Exception as e:
            error_msg = f"실제 RAG 분석 단계 실패: {e}"
            self.pipeline_results["errors"].append(error_msg)
            raise Exception(error_msg)

    def _calculate_rag_confidence(self, industries: List[Dict], past_issues: List[Dict]) -> float:
        """RAG 분석 신뢰도 계산 (실제 점수 기반)"""
        if not industries or not past_issues:
            return 0.0
        
        # 실제 final_score 기반 신뢰도 계산
        industry_avg = sum(ind.get("final_score", 0) for ind in industries) / len(industries)
        past_avg = sum(issue.get("final_score", 0) for issue in past_issues) / len(past_issues)
        
        return round((industry_avg + past_avg) / 2, 1)

    def _save_rag_results(self, rag_result: Dict) -> str:
        """RAG 분석 결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            filename = f"{timestamp}_RealRAG_Enhanced_5issues.json"
            filepath = self.data_dir / filename
            
            save_data = {
                **rag_result,
                "file_info": {
                    "filename": filename,
                    "created_at": datetime.now().isoformat(),
                    "rag_version": "RealRAG_Enhanced_Pipeline_v2.0",
                    "ai_model": "gpt-4o-mini"
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 실제 RAG 분석 결과 저장: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"⚠️ RAG 결과 저장 실패: {e}")
            return ""

    def _prepare_api_data(self, rag_result: Dict) -> Dict:
        """Step 4: API용 데이터 준비 (실제 RAG 결과 포함)"""
        print(f"\n{'='*60}")
        print(f"🌐 Step 4: API 응답 데이터 준비 (실제 RAG 포함)")
        print(f"{'='*60}")
        
        try:
            selected_issues = rag_result.get("selected_issues", [])
            
            # API 응답 형태로 데이터 변환
            api_data = {
                "success": True,
                "data": {
                    "total_crawled": rag_result.get("original_issues_count", 0),
                    "selected_count": len(selected_issues),
                    "selection_criteria": "주식시장 영향도 + 실제 RAG 분석",
                    "selected_issues": []
                },
                "metadata": {
                    "crawled_at": rag_result.get("filter_metadata", {}).get("filtered_at", ""),
                    "categories_processed": getattr(BigKindsCrawler, 'TARGET_CATEGORIES', []),
                    "ai_filter_applied": True,
                    "real_rag_analysis_applied": True,
                    "filter_model": "gpt-4o-mini",
                    "rag_model": "gpt-4o-mini",
                    "rag_confidence": self._calculate_overall_rag_confidence(selected_issues)
                }
            }
            
            # 이슈 데이터 변환 (실제 RAG 결과 포함)
            for issue in selected_issues:
                api_issue = {
                    "이슈번호": issue.get("이슈번호", 0),
                    "제목": issue.get("제목", ""),
                    "내용": issue.get("원본내용", issue.get("내용", "")),
                    "카테고리": issue.get("카테고리", ""),
                    "추출시간": issue.get("추출시간", ""),
                    "주식시장_관련성_점수": issue.get("종합점수", 0),
                    "순위": issue.get("rank", 0),
                    
                    # 실제 RAG 분석 결과 추가
                    "관련산업": issue.get("관련산업", []),
                    "관련과거이슈": issue.get("관련과거이슈", []),
                    "RAG분석신뢰도": issue.get("RAG분석신뢰도", 0.0),
                }
                api_data["data"]["selected_issues"].append(api_issue)
            
            # API 데이터 정렬 (순위별)
            api_data["data"]["selected_issues"].sort(key=lambda x: x.get("순위", 999))
            
            print(f"✅ Step 4 완료: 실제 RAG 분석 포함 API 응답 데이터 준비")
            self.pipeline_results["steps_completed"].append("api_preparation")
            
            return api_data
            
        except Exception as e:
            error_msg = f"API 데이터 준비 실패: {e}"
            self.pipeline_results["errors"].append(error_msg)
            raise Exception(error_msg)

    def _calculate_overall_rag_confidence(self, selected_issues: List[Dict]) -> float:
        """전체 RAG 분석 신뢰도 계산"""
        if not selected_issues:
            return 0.0
        
        confidences = [issue.get("RAG분석신뢰도", 0.0) for issue in selected_issues]
        return round(sum(confidences) / len(confidences), 2)

    def _check_recent_crawling_data(self, max_age_hours: int = 6) -> Optional[Dict]:
        """최근 크롤링 데이터가 있는지 확인"""
        try:
            json_files = list(self.data_dir.glob("*_MultiCategory_*issues.json"))
            if not json_files:
                return None
            
            # 가장 최신 파일 확인
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
            
            # 지정된 시간보다 오래된 경우 새 크롤링 필요
            if file_age > timedelta(hours=max_age_hours):
                print(f"📅 기존 데이터가 {file_age}만큼 오래되어 새 크롤링이 필요합니다.")
                return None
            
            # 기존 데이터 로드
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"♻️ {file_age}만큼 오래된 크롤링 데이터를 재사용합니다.")
            return data
            
        except Exception as e:
            print(f"⚠️ 기존 데이터 확인 실패: {e}")
            return None

    def _save_pipeline_results(self) -> str:
        """파이프라인 실행 결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            filename = f"{timestamp}_RealRAG_Pipeline_Results.json"
            filepath = self.data_dir / filename
            
            save_data = {
                **self.pipeline_results,
                "file_info": {
                    "filename": filename,
                    "created_at": datetime.now().isoformat(),
                    "pipeline_version": "RealRAG_IntegratedNewsPipeline_v2.0",
                    "ai_model": "gpt-4o-mini"
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 파이프라인 결과 저장: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")
            return ""

    def _print_pipeline_summary(self):
        """파이프라인 실행 결과 요약 출력"""
        print(f"\n{'='*80}")
        print(f"🎉 실제 RAG 통합 뉴스 분석 파이프라인 완료!")
        print(f"{'='*80}")
        
        results = self.pipeline_results
        print(f"🆔 파이프라인 ID: {results['pipeline_id']}")
        print(f"⏰ 실행 시간: {results['execution_time']}")
        print(f"📊 최종 상태: {results['final_status']}")
        print(f"🤖 AI 모델: GPT-4o-mini")
        
        print(f"\n✅ 완료된 단계:")
        for step in results["steps_completed"]:
            step_names = {
                "crawling_new": "🕷️ 새 크롤링 실행",
                "crawling_reused": "♻️ 기존 크롤링 데이터 재사용",
                "filtering": "🤖 AI 필터링",
                "real_rag_analysis": "🔍 실제 RAG 분석 (산업 + 과거 이슈)",
                "api_preparation": "🌐 API 데이터 준비"
            }
            print(f"   • {step_names.get(step, step)}")
        
        # 주요 결과 출력
        if results.get("api_ready_data"):
            api_data = results["api_ready_data"]
            print(f"\n📈 최종 결과:")
            print(f"   • 크롤링된 총 이슈: {api_data['data']['total_crawled']}개")
            print(f"   • AI 선별 이슈: {api_data['data']['selected_count']}개")
            print(f"   • 실제 RAG 분석 신뢰도: {api_data['metadata']['rag_confidence']}")
            
            # TOP 3 이슈 미리보기 (실제 RAG 결과 포함)
            selected_issues = api_data["data"]["selected_issues"]
            if selected_issues:
                print(f"\n🏆 TOP 3 선별 이슈 (실제 RAG 분석 포함):")
                for issue in selected_issues[:3]:
                    print(f"   {issue['순위']}. [{issue['카테고리']}] {issue['제목'][:50]}...")
                    print(f"      💰 관련성 점수: {issue['주식시장_관련성_점수']}")
                    
                    industries = issue.get('관련산업', [])
                    past_issues = issue.get('관련과거이슈', [])
                    
                    if industries:
                        industry_names = [ind.get('name', 'N/A') for ind in industries]
                        print(f"      🏭 관련 산업: {', '.join(industry_names)}")
                    else:
                        print(f"      🏭 관련 산업: 없음")
                        
                    if past_issues:
                        past_names = [past.get('name', 'N/A') for past in past_issues]
                        print(f"      📚 관련 과거이슈: {', '.join(past_names)}")
                    else:
                        print(f"      📚 관련 과거이슈: 없음")
                        
                    print(f"      🔍 RAG 신뢰도: {issue.get('RAG분석신뢰도', 0.0)}")
        
        # 에러가 있다면 표시
        if results.get("errors"):
            print(f"\n⚠️ 발생한 오류:")
            for error in results["errors"]:
                print(f"   • {error}")

    def get_latest_api_data(self) -> Optional[Dict]:
        """최신 파이프라인 실행 결과에서 API 데이터 추출"""
        try:
            json_files = list(self.data_dir.glob("*_RealRAG_Pipeline_Results.json"))
            if not json_files:
                # RAG 결과가 없으면 기존 파이프라인 결과 확인
                json_files = list(self.data_dir.glob("*_Pipeline_Results.json"))
            
            if not json_files:
                return None
            
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get("api_ready_data")
            
        except Exception as e:
            print(f"❌ API 데이터 로드 실패: {e}")
            return None

    def run_quick_update(self, force_crawling: bool = False) -> Dict:
        """빠른 업데이트 실행 (실제 RAG 분석 포함)"""
        print("🚀 빠른 업데이트 실행 (실제 RAG + GPT-4o-mini)...")
        
        try:
            return self.run_full_pipeline(
                issues_per_category=10,
                target_filtered_count=5,
                force_new_crawling=force_crawling
            )
        except Exception as e:
            print(f"❌ 빠른 업데이트 실패: {e}")
            # 실패 시 기존 데이터라도 반환 시도
            existing_data = self.get_latest_api_data()
            if existing_data:
                print("♻️ 기존 API 데이터 반환")
                return {"api_ready_data": existing_data, "final_status": "fallback"}
            raise

# 편의 함수들
def run_full_news_pipeline_with_rag(headless: bool = True, 
                                   force_new_crawling: bool = False,
                                   issues_per_category: int = 10) -> Dict:
    """실제 RAG 분석 포함 전체 뉴스 파이프라인 실행 편의 함수"""
    pipeline = IntegratedNewsPipeline(headless=headless)
    return pipeline.run_full_pipeline(
        issues_per_category=issues_per_category,
        force_new_crawling=force_new_crawling
    )

def get_latest_rag_enhanced_issues_for_api():
    """실제 RAG 분석이 포함된 최신 이슈 데이터 반환"""
    try:
        # 1. 실제 RAG Enhanced 파일 직접 찾기
        pipeline = IntegratedNewsPipeline(headless=True)
        data_dir = pipeline.data_dir
        
        # 실제 RAG Enhanced JSON 파일들 찾기
        real_rag_files = list(data_dir.glob("*_RealRAG_Enhanced_*issues.json"))
        
        if real_rag_files:
            # 가장 최신 파일 선택
            latest_file = max(real_rag_files, key=lambda f: f.stat().st_mtime)
            print(f"✅ 실제 RAG 분석 데이터 발견: {latest_file.name}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # selected_issues 반환 (실제 RAG 결과 포함)
            selected_issues = data.get('selected_issues', [])
            print(f"📊 실제 RAG 분석된 {len(selected_issues)}개 이슈 반환")
            return selected_issues
            
        # 2. 실제 RAG 데이터가 없으면 기존 StockFiltered 데이터 확인
        print("🔍 기존 필터링 데이터 확인 중...")
        filtered_files = list(data_dir.glob("*_StockFiltered_*issues.json"))
        
        if filtered_files:
            latest_file = max(filtered_files, key=lambda f: f.stat().st_mtime)
            print(f"📊 기존 필터링 데이터 발견: {latest_file.name}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            selected_issues = data.get('selected_issues', [])
            print(f"⚠️ RAG 분석 없는 {len(selected_issues)}개 이슈 반환 (기본 필터링만)")
            return selected_issues
            
        # 3. 필터링 데이터도 없으면 원본 크롤링 데이터 확인
        print("🔍 원본 크롤링 데이터 확인 중...")
        try:
            from crawling_bigkinds import BigKindsCrawler
            crawler = BigKindsCrawler()
            raw_data = crawler.load_latest_results()
            
            if raw_data and raw_data.get('all_issues'):
                print(f"📊 원본 데이터 {len(raw_data['all_issues'])}개 발견, 상위 5개 반환")
                # 원본 데이터에서 상위 5개만 반환 (임시)
                issues = raw_data['all_issues'][:5]
                return [{
                    "이슈번호": issue.get("이슈번호", i+1),
                    "제목": issue.get("제목", ""),
                    "내용": issue.get("내용", ""),
                    "카테고리": issue.get("카테고리", ""),
                    "추출시간": issue.get("추출시간", ""),
                    "순위": i+1,
                    "관련산업": [],  # 빈 배열
                    "관련과거이슈": [],  # 빈 배열
                    "RAG분석신뢰도": 0.0
                } for i, issue in enumerate(issues)]
        except Exception as e:
            print(f"⚠️ 원본 데이터 로드 실패: {e}")
            
        # 4. 아무 데이터도 없으면 더미 데이터
        print("⚠️ 데이터 없음, 더미 데이터 반환")
        return [
            {
                "이슈번호": 1,
                "제목": "SK하이닉스 AI 반도체 수요 급증",
                "내용": "SK하이닉스가 AI 반도체 수요 증가로 분기 최대 실적을 달성했습니다...",
                "카테고리": "경제",
                "순위": 1,
                "추출시간": datetime.now().isoformat(),
                "주식시장_관련성_점수": 9.2,
                "관련산업": [
                    {
                        "name": "반도체",
                        "final_score": 9.1,
                        "ai_reason": "AI 반도체 수요와 직접적 연관",
                        "vector_similarity": 91.2
                    }
                ],
                "관련과거이슈": [
                    {
                        "name": "2022년 반도체 공급난",
                        "final_score": 8.8,
                        "ai_reason": "반도체 수급 불균형 패턴 유사",
                        "vector_similarity": 88.5,
                        "period": "2022.03 ~ 2022.12"
                    }
                ],
                "RAG분석신뢰도": 8.9
            },
            {
                "이슈번호": 2,
                "제목": "현대차 전기차 배터리 기술 혁신",
                "내용": "현대차가 차세대 배터리 기술 개발로 전기차 시장 선도를 목표로 하고 있습니다...",
                "카테고리": "경제",
                "순위": 2,
                "추출시간": datetime.now().isoformat(),
                "주식시장_관련성_점수": 8.7,
                "관련산업": [
                    {
                        "name": "자동차",
                        "final_score": 8.9,
                        "ai_reason": "전기차 기술 혁신과 직접적 연관",
                        "vector_similarity": 89.3
                    }
                ],
                "관련과거이슈": [
                    {
                        "name": "2021년 전기차 보급 확산",
                        "final_score": 8.2,
                        "ai_reason": "전기차 시장 성장 패턴 유사",
                        "vector_similarity": 82.4,
                        "period": "2021.01 ~ 2021.12"
                    }
                ],
                "RAG분석신뢰도": 8.5
            }
        ]
        
    except Exception as e:
        print(f"실제 RAG 파이프라인 오류: {e}")
        # 에러 발생시 기본 더미 데이터 반환
        return []

def quick_refresh_rag_news_data(force_crawling: bool = False) -> Dict:
    """실제 RAG 분석 포함 뉴스 데이터 빠른 새로고침"""
    pipeline = IntegratedNewsPipeline(headless=True)
    return pipeline.run_quick_update(force_crawling=force_crawling)

# 스케줄링용 함수
def scheduled_daily_update_with_rag():
    """실제 RAG 분석 포함 일일 정기 업데이트용 함수"""
    print("📅 실제 RAG 분석 포함 일일 정기 업데이트 시작...")
    try:
        result = run_full_news_pipeline_with_rag(
            headless=True,
            force_new_crawling=True,  # 정기 업데이트는 항상 새 크롤링
            issues_per_category=10
        )
        print("✅ 실제 RAG 분석 포함 일일 정기 업데이트 완료")
        return result
    except Exception as e:
        print(f"❌ 실제 RAG 분석 포함 일일 정기 업데이트 실패: {e}")
        raise

# 메인 실행
if __name__ == "__main__":
    print("🔄 Real RAG Enhanced Integrated News Analysis Pipeline")
    print("="*80)
    
    # 실행 모드 선택
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "quick":
            print("⚡ 빠른 업데이트 모드 (실제 RAG + GPT-4o-mini)")
            result = quick_refresh_rag_news_data()
            
        elif mode == "force":
            print("🔄 강제 새 크롤링 모드 (실제 RAG + GPT-4o-mini)")
            result = run_full_news_pipeline_with_rag(force_new_crawling=True)
            
        elif mode == "daily":
            print("📅 일일 정기 업데이트 모드 (실제 RAG + GPT-4o-mini)")
            result = scheduled_daily_update_with_rag()
            
        elif mode == "api":
            print("🌐 실제 RAG API 데이터 조회 모드")
            api_data = get_latest_rag_enhanced_issues_for_api()
            if api_data:
                print("✅ 실제 RAG API 데이터 조회 성공")
                print(f"📊 실제 RAG 분석된 이슈 수: {len(api_data)}개")
                for issue in api_data[:3]:  # TOP 3만 미리보기
                    print(f"   • {issue.get('제목', 'N/A')[:50]}...")
                    print(f"     관련산업: {len(issue.get('관련산업', []))}개")
                    print(f"     관련과거이슈: {len(issue.get('관련과거이슈', []))}개")
                    print(f"     RAG 신뢰도: {issue.get('RAG분석신뢰도', 0.0)}")
            else:
                print("❌ 실제 RAG API 데이터 없음")
            sys.exit(0)
            
        else:
            print(f"❌ 알 수 없는 모드: {mode}")
            print("사용법: python integrated_pipeline.py [quick|force|daily|api]")
            sys.exit(1)
    else:
        # 기본 모드: 실제 RAG 분석 포함 파이프라인 실행
        print("🔄 기본 실제 RAG 파이프라인 모드")
        
        pipeline = IntegratedNewsPipeline(headless=False)  # 테스트용으로 브라우저 표시
        
        try:
            result = pipeline.run_full_pipeline(
                issues_per_category=10,
                target_filtered_count=5,
                force_new_crawling=False
            )
            
            print(f"\n🎯 실제 RAG 파이프라인 실행 완료!")
            print(f"   • 상태: {result['final_status']}")
            print(f"   • 실행 시간: {result['execution_time']}")
            print(f"   • AI 모델: GPT-4o-mini")
            
            if result.get("api_ready_data"):
                api_data = result["api_ready_data"]
                print(f"   • 최종 선별 이슈: {api_data['data']['selected_count']}개")
                print(f"   • 실제 RAG 분석 신뢰도: {api_data['metadata']['rag_confidence']}")
                
        except Exception as e:
            print(f"❌ 실제 RAG 파이프라인 실행 실패: {e}")
            traceback.print_exc()
            sys.exit(1)