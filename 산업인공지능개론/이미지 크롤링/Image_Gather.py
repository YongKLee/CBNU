from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


#keyword = '주걱'
keyword = 'laptop'
createFolder('./' + keyword + '_img_download')

chromedriver = 'C://chromedriver.exe'
driver = webdriver.Chrome(chromedriver)
driver.implicitly_wait(3)

# =============================================================================
# 구글 이미지 검색 접속 및 검색어 입력
# =============================================================================
print(keyword, '검색')
driver.get('https://www.google.co.kr/imghp?hl=ko')

Keyword = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
Keyword.send_keys(keyword)

driver.find_element_by_xpath('//*[@id="sbtc"]/button').click()

# =============================================================================
# 스크롤
# =============================================================================
print(keyword + ' 스크롤 중 .............')
elem = driver.find_element_by_tag_name("body")
for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)

try:
    driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[4]/div[2]/input').click()
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)
except:
    pass

# =============================================================================
# 이미지 개수
# =============================================================================
links = []
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
for image in images:
    if image.get_attribute('src') != None:
        links.append(image.get_attribute('src'))

print(keyword + ' 찾은 이미지 개수:', len(links))
time.sleep(2)

# =============================================================================
# 이미지 다운로드
# =============================================================================
for k, i in enumerate(links):
    url = i
    start = time.time()
    urllib.request.urlretrieve(url, "./" + keyword + "_img_download/" + keyword + "_" + str(k) + ".jpg")
    print(str(k + 1) + '/' + str(len(links)) + ' ' + keyword + ' 다운로드 중....... Download time : ' + str(
        time.time() - start)[:5] + ' 초')
print(keyword + ' ---다운로드 완료---')

driver.close()
