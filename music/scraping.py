from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

download_path = "C:\\Users\\User\\htmlcss\\music"

chrome_options = Options()
chrome_options.add_experimental_option('prefs', {
    "download.default_directory": download_path,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

s = Service('C:\\chromedriver.exe')
driver = webdriver.Chrome(service=s, options=chrome_options)

try:
    while True:
        driver.get('https://pixabay.com/music/search/?order=ec')

        wait = WebDriverWait(driver, 10)
        download_button = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, '.buttonBase--r4opq.secondaryButton--xk9cO.base--jzyee.light--uBcBI')))
        download_button.click()

        close_button = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, '.close--+Fj1i')))
        close_button.click()

        time.sleep(20)

        next_page_link = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, 'a.buttonBase--r4opq.secondaryButton--xk9cO.base--jzyee.light--uBcBI[href]')))
        next_page_link.click()

except Exception as e:
    print(e)
finally:
    driver.quit()
