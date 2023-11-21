from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

download_path = "C:\\Users\\User\\htmlcss\\music"

chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('prefs', {
    "download.default_directory": download_path,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

s = Service('C:\\chromedriver.exe')
driver = webdriver.Chrome(service=s, options=chrome_options)

try:
    driver.get('https://ncs.io/music')
    while True:
        elements = driver.find_elements(By.CSS_SELECTOR, 'div.img')
        for element in elements:
            element.click()
            download_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, 'a.btn.black.panel-btn'))
            )
            download_button.click()

            actual_download_link = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, 'a[href^="/track/download/"]'))
            )
            actual_download_link.click()

            time.sleep(15)
            driver.back()

        next_page_link = driver.find_element(By.CSS_SELECTOR, 'a.page-link')
        if next_page_link:
            next_page_link.click()
        else:
            break

except Exception as e:
    print(e)
finally:
    driver.quit()
