#!/usr/bin/env python3
"""
BigKinds Crawler - Enhanced Multi-Category Version
7개 카테고리에서 각각 10개씩 총 70개 이슈 크롤링 + 기존 호환성 유지
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import traceback
import json
from datetime import datetime
import os
from pathlib import Path
from typing import List, Dict, Optional

class BigKindsCrawler:
    """BigKinds 다중 카테고리 크롤러"""
    
    # 크롤링 대상 카테고리 (전체 제외)
    TARGET_CATEGORIES = ["정치", "경제", "사회", "문화", "국제", "지역", "IT과학"]
    
    def __init__(self, data_dir: str = "data2", headless: bool = False, issues_per_category: int = 10):
        """
        크롤러 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
            headless: 헤드리스 모드 실행 여부
            issues_per_category: 카테고리별 크롤링할 이슈 수
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.headless = headless
        self.issues_per_category = issues_per_category
        self.driver = None
        self.wait = None
        
        # 크롤링 결과 저장
        self.crawling_results = {
            "total_issues": 0,
            "categories": {},
            "crawling_log": [],
            "crawled_at": "",
            "all_issues": [],  
        }
        
        # 크롬 옵션 설정
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--headless')  # 헤드리스 모드
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        

    def crawl_current_issues(self, category: str = "전체", max_issues: int = 10) -> Dict:
        """
        기존 호환성을 위한 메서드
        
        Args:
            category: 크롤링할 카테고리 ("전체"면 모든 카테고리 크롤링)
            max_issues: 크롤링할 최대 이슈 수
            
        Returns:
            크롤링 결과 딕셔너리
        """
        if category == "전체":
            # 새로운 다중 카테고리 크롤링 실행
            return self.crawl_all_categories()
        else:
            # 단일 카테고리 크롤링 (기존 방식과 유사)
            return self._crawl_single_category(category, max_issues)

    def crawl_all_categories(self) -> Dict:
        """
        모든 카테고리에서 이슈 크롤링
        
        Returns:
            크롤링 결과 딕셔너리
        """
        start_time = datetime.now()
        self.crawling_results["crawled_at"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"🚀 다중 카테고리 크롤링 시작")
        print(f"📋 대상 카테고리: {', '.join(self.TARGET_CATEGORIES)}")
        print(f"📊 카테고리별 이슈 수: {self.issues_per_category}개")
        print(f"🎯 예상 총 이슈 수: {len(self.TARGET_CATEGORIES) * self.issues_per_category}개")
        
        try:
            self._setup_driver()
            self._navigate_to_bigkinds()
            
            # 각 카테고리별 크롤링
            for idx, category in enumerate(self.TARGET_CATEGORIES, 1):
                print(f"\n{'='*60}")
                print(f"📂 [{idx}/{len(self.TARGET_CATEGORIES)}] '{category}' 카테고리 크롤링 시작")
                print(f"{'='*60}")
                
                try:
                    category_issues = self._crawl_category(category)
                    self.crawling_results["categories"][category] = category_issues
                    self.crawling_results["all_issues"].extend(category_issues)
                    
                    print(f"✅ '{category}' 카테고리 완료: {len(category_issues)}개 이슈")
                    self.crawling_results["crawling_log"].append({
                        "category": category,
                        "status": "success",
                        "issues_count": len(category_issues),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # 카테고리 간 대기 (사이트 부하 방지)
                    if idx < len(self.TARGET_CATEGORIES):
                        print("⏳ 다음 카테고리 대기 중... (3초)")
                        time.sleep(3)
                    
                except Exception as e:
                    error_msg = f"❌ '{category}' 카테고리 크롤링 실패: {e}"
                    print(error_msg)
                    self.crawling_results["crawling_log"].append({
                        "category": category,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    # 한 카테고리 실패해도 다음 카테고리 계속 진행
                    continue
            
            # 결과 정리
            self.crawling_results["total_issues"] = len(self.crawling_results["all_issues"])
            
            # 결과 저장
            saved_file = self._save_results()
            
            # 최종 결과 출력
            self._print_summary()
            
            return {
                **self.crawling_results,
                "saved_file": saved_file,
                "execution_time": str(datetime.now() - start_time)
            }
            
        except Exception as e:
            print(f"❌ 전체 크롤링 실패: {e}")
            traceback.print_exc()
            raise
        finally:
            self._cleanup_driver()

    def _crawl_single_category(self, category: str, max_issues: int) -> Dict:
        """단일 카테고리 크롤링 (기존 방식 호환)"""
        print(f"🎯 단일 카테고리 크롤링: {category}")
        
        try:
            self._setup_driver()
            self._navigate_to_bigkinds()
            
            category_issues = self._crawl_category(category, max_issues)
            
            result = {
                "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_issues": len(category_issues),
                "source": "bigkinds.or.kr",
                "category": category,
                "issues": category_issues
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 단일 카테고리 크롤링 실패: {e}")
            raise
        finally:
            self._cleanup_driver()

    def _setup_driver(self):
        """Chrome 드라이버 설정 - 자동화 탐지 방지 기능 포함"""
        options = webdriver.ChromeOptions()
        
        if self.headless:
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
        else:
            options.add_argument("--start-maximized")
        
        # 자동화 탐지 방지
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # 추가 안정성 설정
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")

        self.driver = webdriver.Chrome(options=options)
        # webdriver 속성 숨기기
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 15)  # 대기 시간 증가
        
        print("✅ Chrome 드라이버 설정 완료")

    def _cleanup_driver(self):
        """드라이버 안전 종료"""
        if self.driver:
            try:
                self.driver.quit()
                print("✅ 드라이버 종료 완료")
            except Exception as e:
                print(f"⚠️ 드라이버 종료 중 오류: {e}")
            finally:
                self.driver = None
                self.wait = None

    def _navigate_to_bigkinds(self):
        """BigKinds 사이트 접속 및 초기 설정"""
        print("🌐 BigKinds 사이트 접속 중...")
        
        try:
            self.driver.get("https://www.bigkinds.or.kr/")
            time.sleep(3)  # 페이지 로딩 대기
            
            # 팝업 닫기 (있다면)
            try:
                popup_close = self.driver.find_element(By.CSS_SELECTOR, ".popup-close-btn")
                popup_close.click()
            except:
                pass
                
            print("✅ BigKinds 사이트 접속 완료")
            return True
            
        except Exception as e:
            print(f"❌ 사이트 접속 실패: {str(e)}")
            return False

    def _scroll_to_issues_section(self):
        """오늘의 이슈 섹션으로 스크롤"""
        try:
            self.driver.execute_script("window.scrollTo(0, 880);")
            time.sleep(2)
            print("✅ 이슈 섹션 스크롤 완료")
        except Exception as e:
            print(f"⚠️ 스크롤 실패: {e}")

    def _crawl_category(self, category: str, max_issues: int = None) -> List[Dict]:
        """
        특정 카테고리의 이슈들 크롤링
        
        Args:
            category: 크롤링할 카테고리명
            max_issues: 최대 이슈 수 (None이면 기본값 사용)
            
        Returns:
            해당 카테고리의 이슈 리스트
        """
        if max_issues is None:
            max_issues = self.issues_per_category
            
        try:
            # 카테고리 선택 및 스크롤
            self._scroll_to_issues_section() 
            self._click_category(category)
            
            # 해당 카테고리의 이슈들 크롤링
            issues = self._crawl_issues_in_category(category, max_issues)
            
            return issues
            
        except Exception as e:
            print(f"❌ 카테고리 '{category}' 크롤링 실패: {e}")
            raise

    def _click_category(self, category: str):
        """카테고리 선택"""
        try:
            print(f"🎯 '{category}' 카테고리 선택 중...")
            
            # 카테고리 버튼 찾기 및 클릭
            category_selector = f'a.issue-category[data-category="{category}"]'
            category_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, category_selector))
            )
            
            # JavaScript로 클릭 (더 안정적)
            self.driver.execute_script("arguments[0].click();", category_button)
            time.sleep(4)  # 카테고리 변경 대기 시간 증가
            
            print(f"✅ '{category}' 카테고리 선택 완료")
            
        except Exception as e:
            print(f"❌ 카테고리 '{category}' 선택 실패: {e}")
            print("사용 가능한 카테고리: 정치, 경제, 사회, 문화, 국제, 지역, IT과학")
            raise

    def _select_category(self, category: str) -> bool:
        try:
            # 카테고리 메뉴 클릭
            menu_btn = self.driver.find_element(By.CSS_SELECTOR, ".category-menu-btn")
            menu_btn.click()
            time.sleep(1)
            
            # 카테고리 선택
            category_btn = self.driver.find_element(By.XPATH, 
                f"//div[contains(@class, 'category-item') and contains(text(), '{category}')]")
            category_btn.click()
            time.sleep(2)
            
            print(f"✅ '{category}' 카테고리 선택 완료")
            return True
            
        except Exception as e:
            print(f"❌ 카테고리 '{category}' 선택 실패: {str(e)}")
            print(f"\n사용 가능한 카테고리: {', '.join(self.TARGET_CATEGORIES)}")
            return False

    def _crawl_issues_in_category(self, category: str, max_issues: int) -> List[Dict]:
        """카테고리 내 이슈들 크롤링"""
        issues = []
        
        for i in range(1, max_issues + 1):
            print(f"  📰 [{i}/{max_issues}] 이슈 처리 중...")
            
            try:
                # 3번째 이슈부터는 슬라이드 넘기기 필요
                if i >= 3:
                    self._navigate_slides(i)
                
                # 이슈 데이터 추출
                issue_data = self._extract_issue_data(i, category)
                print(f"📌 DEBUG: crawling_results keys: {list(self.crawling_results.keys())}")
                if issue_data:
                    issues.append(issue_data)
                    print(f"    ✅ 이슈 {i} 추출 완료: {issue_data['제목'][:30]}...")
                
                # 팝업 닫기 및 위치 복원
                self._close_popup_and_restore()
                
                # 이슈 간 짧은 대기
                time.sleep(1)
                
            except Exception as e:
                print(f"    ❌ 이슈 {i} 처리 실패: {e}")
                # 개별 이슈 실패해도 다음 이슈 계속 진행
                continue
        
        return issues

    def _navigate_slides(self, issue_num: int):
        """슬라이드 넘기기 (3번째 이슈부터 필요)"""
        slides_to_move = issue_num - 3
        
        for slide in range(slides_to_move):
            try:
                next_btn = self.driver.find_element(
                    By.CSS_SELECTOR, 
                    'div.swiper-button-next.section2-btn.st2-sw1-next'
                )
                
                # 버튼이 비활성화되었는지 확인
                is_disabled = next_btn.get_attribute('aria-disabled') == 'true'
                if is_disabled:
                    print(f"    ⚠️ 슬라이드 끝에 도달 (이슈 {issue_num})")
                    break
                
                self.driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(1)
                
            except Exception as e:
                print(f"    ⚠️ 슬라이드 넘기기 실패 (이슈 {issue_num}): {e}")
                break

    def _extract_issue_data(self, issue_num: int, category: str) -> Optional[Dict]:
        """개별 이슈 데이터 추출"""
        try:
            # 이슈 클릭
            issue_selector = f'div.swiper-slide:nth-child({issue_num}) .issue-item-link'
            issue_element = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, issue_selector))
            )
            
            # 요소가 보이도록 스크롤 후 클릭
            self.driver.execute_script("arguments[0].scrollIntoView(true);", issue_element)
            time.sleep(0.5)
            self.driver.execute_script("arguments[0].click();", issue_element)
            
            # 팝업 내용 추출
            title_elem = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'p.issuPopTitle'))
            )
            content_elem = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'p.pT20.issuPopContent'))
            )

            title = title_elem.text.strip()
            content = content_elem.text.strip()
            
            # 유니크 ID 생성 (카테고리 + 순서)
            unique_id = f"{category}_{issue_num}"

            return {
                "이슈번호": len(self.crawling_results["all_issues"]) + 1,  # 전체 순서
                "카테고리별_번호": issue_num,  # 카테고리 내 순서
                "카테고리": category,
                "제목": title,
                "내용": content,
                "추출시간": datetime.now().isoformat(),
                "고유ID": unique_id
            }
            
        except Exception as e:
            print(f"    ❌ 이슈 {issue_num} 데이터 추출 실패: {e}")
            print(f"   ↳ 예외 타입: {type(e).__name__}")
            print(f"   ↳ 메시지: {str(e)}")
            traceback.print_exc()
            return None

    def _close_popup_and_restore(self):
        """팝업 닫기 및 스크롤 위치 복원"""
        try:
            # ESC로 팝업 닫기
            ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)
            
            # 다시 이슈 섹션으로 스크롤
            self.driver.execute_script("window.scrollTo(0, 880);")
            time.sleep(1)
            
        except Exception as e:
            print(f"    ⚠️ 팝업 닫기 실패: {e}")

    def _save_results(self) -> str:
        """크롤링 결과를 JSON 파일로 저장"""
        try:
            # 타임스탬프와 총 이슈 수를 포함한 파일명
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            filename = f"{timestamp}_MultiCategory_{self.crawling_results['total_issues']}issues.json"
            filepath = self.data_dir / filename
            
            # 저장할 데이터 구조
            save_data = {
                **self.crawling_results,
                "file_info": {
                    "filename": filename,
                    "created_at": datetime.now().isoformat(),
                    "crawler_version": "BigKindsCrawler_v1.0"
                }
            }
            
            # JSON 파일로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 크롤링 결과 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
            raise

    def _print_summary(self):
        """크롤링 결과 요약 출력"""
        print(f"\n{'='*80}")
        print(f"🎉 다중 카테고리 크롤링 완료!")
        print(f"{'='*80}")
        print(f"📊 전체 통계:")
        print(f"   • 처리된 카테고리: {len(self.crawling_results['categories'])}/{len(self.TARGET_CATEGORIES)}개")
        print(f"   • 총 수집 이슈: {self.crawling_results['total_issues']}개")
        print(f"   • 크롤링 시간: {self.crawling_results['crawled_at']}")
        
        print(f"\n📂 카테고리별 결과:")
        for category, issues in self.crawling_results["categories"].items():
            print(f"   • {category}: {len(issues)}개 이슈")
        
        # 실패한 카테고리가 있다면 표시
        failed_categories = [log for log in self.crawling_results["crawling_log"] if log["status"] == "failed"]
        if failed_categories:
            print(f"\n⚠️ 실패한 카테고리:")
            for failed in failed_categories:
                print(f"   • {failed['category']}: {failed['error']}")
        
        print(f"\n✅ 크롤링 성공률: {len(self.crawling_results['categories'])}/{len(self.TARGET_CATEGORIES)} ({len(self.crawling_results['categories'])/len(self.TARGET_CATEGORIES)*100:.1f}%)")

    def load_latest_results(self) -> Optional[Dict]:
        """최신 크롤링 결과 로드"""
        try:
            json_files = list(self.data_dir.glob("*_MultiCategory_*issues.json"))
            if not json_files:
                print("📂 저장된 다중 카테고리 크롤링 데이터가 없습니다.")
                return None
            
            # 파일 수정 시간을 기준으로 가장 최신 파일 선택
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ 최신 다중 카테고리 데이터 로드: {latest_file.name}")
            print(f"📊 로드된 데이터: {data.get('total_issues', 0)}개 이슈")
            
            return data
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None


# 편의 함수들 (기존 호환성)
def crawl_all_categories_quick(headless: bool = True, issues_per_category: int = 10) -> Dict:
    """빠른 다중 카테고리 크롤링 함수"""
    crawler = BigKindsCrawler(headless=headless, issues_per_category=issues_per_category)
    return crawler.crawl_all_categories()

def load_latest_multi_data() -> Optional[Dict]:
    """최신 다중 카테고리 크롤링 데이터 로드"""
    crawler = BigKindsCrawler()
    return crawler.load_latest_results()


# 메인 실행
if __name__ == "__main__":
    print("🚀 BigKinds Multi-Category Crawler")
    
    # 크롤러 설정
    crawler = BigKindsCrawler(
        headless=False,  # 테스트 시에는 브라우저 보기
        issues_per_category=10  # 카테고리별 10개씩
    )
    
    try:
        # 전체 카테고리 크롤링 실행
        result = crawler.crawl_all_categories()
        
        print(f"\n🎯 최종 결과:")
        print(f"   • 저장 파일: {result['saved_file']}")
        print(f"   • 실행 시간: {result['execution_time']}")
        print(f"   • 총 이슈 수: {result['total_issues']}개")
        
        # 결과 데이터 간단 확인
        if result['all_issues']:
            print(f"\n📰 수집된 이슈 예시:")
            for i, issue in enumerate(result['all_issues'][:3], 1):
                print(f"   {i}. [{issue['카테고리']}] {issue['제목'][:50]}...")
        
    except Exception as e:
        print(f"❌ 크롤링 실행 실패: {e}")
        traceback.print_exc()