#!/usr/bin/env python3
"""
RAG 분석 시스템 모듈 - FastAPI 연동용
현재 뉴스 → 과거 이슈 매칭 → 산업 연결 → 종합 분석
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback
import openai

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleCSVAnalyzer:
    def __init__(self, industry_csv_path, past_issue_csv_path):
        self.industry_df = pd.read_csv(industry_csv_path)
        self.past_df = pd.read_csv(past_issue_csv_path)
        
        # 산업 설명 벡터화
        self.industry_vectorizer = TfidfVectorizer()
        self.industry_tfidf = self.industry_vectorizer.fit_transform(self.industry_df['상세내용'].fillna(''))

        # 과거 뉴스 벡터화
        self.past_vectorizer = TfidfVectorizer()
        self.past_tfidf = self.past_vectorizer.fit_transform(self.past_df['Contents'].fillna(''))

    def analyze(self, news_content, top_k=3):
        # 산업 분석
        input_vec = self.industry_vectorizer.transform([news_content])
        industry_sim = cosine_similarity(input_vec, self.industry_tfidf)[0]
        industry_results = self.industry_df.copy()
        industry_results['score'] = industry_sim
        top_industries = industry_results.sort_values('score', ascending=False).head(top_k)

        # 과거 이슈 분석
        input_vec2 = self.past_vectorizer.transform([news_content])
        past_sim = cosine_similarity(input_vec2, self.past_tfidf)[0]
        past_results = self.past_df.copy()
        past_results['score'] = past_sim
        top_past = past_results.sort_values('score', ascending=False).head(top_k)

        return {
            "related_industries": top_industries[['KRX 업종명', '상세내용', 'score']].rename(
    columns={"KRX 업종명": "name", "상세내용": "description"}
).to_dict(orient='records'),
            "similar_past_issues": top_past[['Issue_name', 'Contents', 'score']].rename(
    columns={"Issue_name": "title", "Contents": "content"}
).to_dict(orient='records')
        }

class RAGAnalyzer:
    """RAG 기반 뉴스 분석 시스템"""
    
    def __init__(self):
        """초기화 및 환경 설정"""
        load_dotenv(override=True)
        
        # 환경 변수 확인
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not self.openai_api_key or not self.pinecone_api_key:
            raise ValueError("OPENAI_API_KEY 또는 PINECONE_API_KEY가 설정되지 않았습니다.")
        
        # 모델 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            max_tokens=4095,
            api_key=self.openai_api_key
        )
        
        self.embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.openai_api_key
        )
        
        # Pinecone 벡터 스토어 초기화
        self._init_vector_stores()
        
        # 프롬프트 템플릿 초기화
        self._init_prompts()
    
    def _init_vector_stores(self):
        """Pinecone 벡터 스토어 초기화"""
        try:
            # 산업 분류 벡터 스토어
            self.industry_store = PineconeVectorStore(
                index_name="lastproject",
                embedding=self.embedding,
                namespace="industry"
            )
            
            # 과거 이슈 벡터 스토어  
            self.past_issue_store = PineconeVectorStore(
                index_name="lastproject",
                embedding=self.embedding,
                namespace="past_issue"
            )
            
            print("✅ Pinecone 벡터 스토어 초기화 완료")
            
        except Exception as e:
            print(f"❌ Pinecone 초기화 실패: {e}")
            raise
    
    def _init_prompts(self):
        """프롬프트 템플릿 초기화"""
        
        # 종합 분석 프롬프트
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 주식 시장에 대한 통찰력 있는 분석을 제공하는 전문 금융 애널리스트입니다.
개인 투자자들이 시장 뉴스를 이해하고 현명한 판단을 내릴 수 있도록 돕는 역할을 합니다.

당신의 목표는 다음과 같습니다:
- 현재 뉴스가 무엇을 의미하는지 설명하고,
- 과거 유사 뉴스와 비교해 변화된 점을 분석하며,  
- 관련 산업의 특성을 바탕으로 어떤 영향이 있을 수 있는지 알려줍니다.

설명은 투자자 입장에서 알기 쉽게 풀어주고, 전문 용어는 피하며, 실제 사례나 비유를 활용해 이해를 돕습니다.
투자 판단에 참고할 만한 시사점이나 주의할 점도 포함해 주세요.

**중요**: 투자 추천이나 매수/매도 권유는 절대 하지 마세요. 교육적 분석만 제공하세요.
            """),
            ("human", """
아래 세 가지 정보를 바탕으로 종합적인 분석을 해주세요.

## 현재 뉴스:
{current_news}

## 과거 유사 이슈:
{past_issues}

## 관련 산업 정보:
{industry_context}

형식: 2~3개의 문단으로, 명확하고 쉽게 설명해 주세요.  
분석은 한국어로 작성해 주세요.
            """)
        ])
        
        # 신뢰도 평가 프롬프트
        self.confidence_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 분석의 신뢰도를 평가하는 전문가입니다.
현재 뉴스와 과거 이슈, 산업 정보 간의 관련성을 0.0~1.0 사이의 점수로 평가하세요.

평가 기준:
- 0.9~1.0: 매우 높은 유사성, 명확한 연관성
- 0.7~0.8: 높은 유사성, 강한 연관성  
- 0.5~0.6: 보통 유사성, 적당한 연관성
- 0.3~0.4: 낮은 유사성, 약한 연관성
- 0.0~0.2: 매우 낮은 유사성, 거의 무관

JSON 형태로 응답하세요: {"confidence": 점수, "reason": "평가 이유"}
            """),
            ("human", """
현재 뉴스: {current_news}
과거 이슈: {past_issues}
관련 산업: {industry_context}
            """)
        ])
    
    async def search_similar_past_issues(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """과거 유사 이슈 검색"""
        try:
            print(f"🔍 과거 유사 이슈 검색: '{query[:50]}...'")
            
            # Pinecone에서 유사 이슈 검색
            similar_docs = self.past_issue_store.similarity_search(
                query, 
                k=top_k,
                namespace="past_issue"
            )
            
            results = []
            for i, doc in enumerate(similar_docs):
                # 문서 내용 파싱
                content_lines = doc.page_content.split('\n')
                
                issue_data = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "parsed_info": self._parse_past_issue_content(content_lines)
                }
                
                results.append(issue_data)
            
            print(f"✅ {len(results)}개 유사 이슈 발견")
            return results
            
        except Exception as e:
            print(f"❌ 과거 이슈 검색 실패: {e}")
            return []
    
    async def search_related_industries(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """관련 산업 검색"""
        try:
            print(f"🏭 관련 산업 검색: '{query[:50]}...'")
            
            # Pinecone에서 관련 산업 검색
            similar_docs = self.industry_store.similarity_search(
                query,
                k=top_k, 
                namespace="industry"
            )
            
            results = []
            for i, doc in enumerate(similar_docs):
                content_lines = doc.page_content.split('\n')
                
                industry_data = {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "parsed_info": self._parse_industry_content(content_lines)
                }
                
                results.append(industry_data)
            
            print(f"✅ {len(results)}개 관련 산업 발견")
            return results
            
        except Exception as e:
            print(f"❌ 관련 산업 검색 실패: {e}")
            return []
    
    def _parse_past_issue_content(self, content_lines: List[str]) -> Dict[str, str]:
        """과거 이슈 내용 파싱"""
        parsed = {
            "issue_name": "미확인",
            "related_industries": "미확인", 
            "industry_reason": "미확인",
            "start_date": "미확인",
            "end_date": "미확인"
        }
        
        for line in content_lines:
            line = line.strip()
            if line.startswith('Issue_name:'):
                parsed["issue_name"] = line.replace('Issue_name:', '').strip()
            elif line.startswith('관련 산업:'):
                parsed["related_industries"] = line.replace('관련 산업:', '').strip()
            elif line.startswith('산업 이유:'):
                parsed["industry_reason"] = line.replace('산업 이유:', '').strip()
            elif line.startswith('Start_date:'):
                parsed["start_date"] = line.replace('Start_date:', '').strip()
            elif line.startswith('Fin_date:'):
                parsed["end_date"] = line.replace('Fin_date:', '').strip()
        
        return parsed
    
    def _parse_industry_content(self, content_lines: List[str]) -> Dict[str, str]:
        """산업 정보 내용 파싱"""
        parsed = {
            "industry_name": "미확인",
            "description": ""
        }
        
        for line in content_lines:
            line = line.strip()
            if line.startswith('KRX 업종명:') or line.startswith('﻿KRX 업종명:'):
                parsed["industry_name"] = line.replace('KRX 업종명:', '').replace('﻿KRX 업종명:', '').strip()
            elif line.startswith('상세내용:'):
                parsed["description"] = line.replace('상세내용:', '').strip()
            elif parsed["description"] and not line.startswith('KRX') and line:
                # 상세내용 continuation
                parsed["description"] += " " + line
        
        return parsed
    
    async def comprehensive_analysis(self, current_news: str, max_past_issues: int = 3, max_industries: int = 3) -> Dict:
        """종합 분석 실행"""
        try:
            print("🧠 종합 분석 시작...")
            
            # 1. 과거 유사 이슈 검색
            print(f"🔍 과거 유사 이슈 검색: '{current_news[:50]}...'")
            past_issues = self.search_past_issues(current_news, top_k=max_past_issues)
            print(f"✅ {len(past_issues)}개 유사 이슈 발견")
            
            # 2. 관련 산업 검색  
            print(f"🏭 관련 산업 검색: '{current_news[:50]}...'")
            industries = self.search_industries(current_news, top_k=max_industries)
            print(f"✅ {len(industries)}개 관련 산업 발견")
            
            # 3. LLM 재랭킹 부분 완전 제거 - 바로 기본 결과 사용
            final_past_issues = past_issues[:max_past_issues]
            final_industries = industries[:max_industries]
            
            # 4. 최종 LLM 분석
            print("🎯 최종 분석 생성 중...")
            explanation = self._generate_explanation(
                current_news, 
                final_past_issues, 
                final_industries
            )
            
            return {
                "explanation": explanation,
                "past_issues": final_past_issues,
                "industries": final_industries,
                "metadata": {
                    "analysis_time": datetime.now().isoformat(),
                    "past_issues_count": len(final_past_issues),
                    "industries_count": len(final_industries)
                }
            }
            
        except Exception as e:
            print(f"❌ 종합 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return {
                "explanation": "분석 중 오류가 발생했습니다.",
                "past_issues": [],
                "industries": [],
                "metadata": {"error": str(e)}
            }
    
    def _format_past_issues_for_analysis(self, past_issues: List[Dict]) -> str:
        if not past_issues:
            return "관련된 과거 이슈를 찾을 수 없습니다."
        
        formatted_texts = []
        for i, issue in enumerate(past_issues[:3]):
            parsed = issue.get("parsed_info", {})
            reason = issue.get("reason", "LLM 판단 이유를 찾을 수 없습니다.")
            
            text = f"""
    [과거 이슈 {i+1}]
    • 이슈명: {parsed.get('issue_name', '미확인')}
    • 관련 산업: {parsed.get('related_industries', '미확인')}
    • 발생 기간: {parsed.get('start_date', '미확인')} ~ {parsed.get('end_date', '미확인')}
    • 선택된 이유 (LLM): {reason}
            """.strip()
            
            formatted_texts.append(text)
        
        return "\n\n".join(formatted_texts)
    
    def _format_industries_for_analysis(self, industries: List[Dict]) -> str:
        if not industries:
            return "관련된 산업 정보를 찾을 수 없습니다."

        formatted_texts = []
        for i, industry in enumerate(industries[:3]):
            parsed = industry.get("parsed_info", {})
            reason = industry.get("reason", "LLM 판단 이유를 찾을 수 없습니다.")

            text = f"""
    [관련 산업 {i+1}]
    • 산업명: {parsed.get('industry_name', '미확인')}
    • 선택된 이유 (LLM): {reason}
            """.strip()

            formatted_texts.append(text)

        return "\n\n".join(formatted_texts)
    
    async def quick_analysis(self, news_content: str) -> Dict[str, Any]:
        """빠른 분석 (간단한 버전)"""
        try:
            # 기본 산업 매칭만 수행
            industries = await self.search_related_industries(news_content, top_k=2)
            
            if not industries:
                return {
                    "explanation": "관련 산업을 찾을 수 없어 분석을 수행할 수 없습니다.",
                    "confidence": 0.0,
                    "industries": []
                }
            
            # 간단한 설명 생성
            industry_names = [ind.get("parsed_info", {}).get("industry_name", "미확인") 
                            for ind in industries]
            
            explanation = f"""
현재 뉴스는 주로 {', '.join(industry_names)} 산업과 관련이 있는 것으로 분석됩니다.

이러한 산업들은 해당 뉴스의 영향을 직간접적으로 받을 수 있으며, 
투자자들은 이러한 산업 동향을 주의 깊게 살펴볼 필요가 있습니다.

더 상세한 분석을 원하시면 전체 분석 기능을 이용해주세요.
            """.strip()
            
            return {
                "explanation": explanation,
                "confidence": 0.6,
                "industries": industry_names
            }
            
        except Exception as e:
            print(f"❌ 빠른 분석 실패: {e}")
            return {
                "explanation": "빠른 분석 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "industries": []
            }

# ===== 테스트 및 직접 실행용 =====

async def test_rag_system():
    """RAG 시스템 테스트"""
    print("🧪 RAG 분석 시스템 테스트")
    
    try:
        # RAG 분석기 초기화
        analyzer = RAGAnalyzer()
        
        # 테스트 뉴스
        test_news = """
SK텔레콤(SKT)이 최근 사이버 보안 사고로 인해 고객들에게 계약 해지 수수료를 면제해 주겠다고 발표했는데, 
이로 인해 많은 고객들이 경쟁사인 KT와 LG U+로 이동하고 있습니다. 
하루에만 6,656명의 순손실을 기록했으며, 일주일 동안 28,566명의 순손실을 보였습니다.
        """
        
        print(f"📝 테스트 뉴스: {test_news[:100]}...")
        
        # 1. 과거 이슈 검색 테스트
        print("\n1️⃣ 과거 이슈 검색 테스트")
        past_issues = await analyzer.search_similar_past_issues(test_news)
        for issue in past_issues:
            parsed = issue["parsed_info"]
            print(f"  • {parsed['issue_name']}")
        
        # 2. 관련 산업 검색 테스트  
        print("\n2️⃣ 관련 산업 검색 테스트")
        industries = await analyzer.search_related_industries(test_news)
        for industry in industries:
            parsed = industry["parsed_info"]
            print(f"  • {parsed['industry_name']}")
        
        # 3. 종합 분석 테스트
        print("\n3️⃣ 종합 분석 테스트")
        analysis = await analyzer.comprehensive_analysis(test_news)
        print(f"신뢰도: {analysis['confidence']:.2f}")
        print(f"분석 결과:\n{analysis['explanation'][:200]}...")
        
        # 4. 빠른 분석 테스트
        print("\n4️⃣ 빠른 분석 테스트")
        quick = await analyzer.quick_analysis(test_news)
        print(f"관련 산업: {quick['industries']}")
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        traceback.print_exc()

from scripts.vector_search import VectorSearchEngine
from langchain_openai import OpenAIEmbeddings

class EmbeddingBasedAnalyzer:
    """OpenAI 임베딩 기반 산업/과거이슈 유사도 분석기"""

    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.embedding_model = OpenAIEmbeddings()

    def analyze(self, content: str, top_k: int = 5) -> Dict[str, Any]:
        # 텍스트 → 임베딩 벡터화
        embedding = self.embedding_model.embed_query(content)

        # 산업/이슈 검색
        industry_matches = self.vector_engine.query_embedding("industry", embedding, top_k)
        issue_matches = self.vector_engine.query_embedding("past_issue", embedding, top_k)

        # 결과 포맷 정제
        related_industries = [
            {
                "name": item["metadata"].get("name", ""),
                "description": item["metadata"].get("description", ""),
                "score": item.get("score", 0.0)
            }
            for item in industry_matches
        ]
        similar_past_issues = [
            {
                "title": item["metadata"].get("title", ""),
                "content": item["metadata"].get("content", ""),
                "score": item.get("score", 0.0)
            }
            for item in issue_matches
        ]

        return {
            "related_industries": related_industries,
            "similar_past_issues": similar_past_issues
                    }

if __name__ == "__main__":
    # 직접 실행 시 테스트
    asyncio.run(test_rag_system())

