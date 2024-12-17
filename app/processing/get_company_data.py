from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import logging
import time
import pandas as pd
from datetime import datetime
import os

# Encapsulate the existing code into a main function
def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set up Firefox options for headless mode
    options = Options()
    options.add_argument("--headless")
    
    # Set up Selenium with Firefox WebDriver
    driver = webdriver.Firefox(options=options)
    
    # Step 1: Fetch the webpage
    url = "https://live.euronext.com/en/markets/oslo/equities/list"
    driver.get(url)
    logging.info("Page loaded")
    
    # Step 1: Wait for the download button to be clickable and click it
    try:
        download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="stocks-data-table-es_wrapper"]/div[1]/div[2]/button'))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
        driver.execute_script("arguments[0].click();", download_button)
        logging.info("Download button clicked")
    except Exception as e:
        logging.error("Download button not found or not clickable: %s", e)

    # Step 2: Wait for the submit button to be clickable and click it
    try:
        submit_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="downloadModal"]/div/div/div[2]/input'))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", submit_button)
        driver.execute_script("arguments[0].click();", submit_button)
        logging.info("Submit button clicked")
    except Exception as e:
        logging.error("Submit button not found or not clickable: %s", e)

    time.sleep(1.5)

    # Get today's date
    today = datetime.today().strftime('%Y-%m-%d')

    # Read the Excel file
    file_path = f'/Users/mattiashaughom/Downloads/Euronext_Equities_{today}.xlsx'
    df = pd.read_excel(file_path)

    # Drop rows 2, 3, and 4 (index 1, 2, 3)
    df = df.drop([0, 1, 2]).reset_index(drop=True)

    # Prepare a DataFrame to store the results
    results = pd.DataFrame(columns=['ISIN', 'Name', 'Symbol', 'Industry', 'Sector'])

    # Iterate through each ISIN
    for index, row in df.iterrows():
        isin = row['ISIN']
        name = row['Name']
        symbol = row['Symbol']
        url = f"https://live.euronext.com/en/product/equities/{isin}-XOSL/market-information"
        driver.get(url)

        # Wait for the main block to load and scroll if necessary
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="fs_icb_block"]'))
        )
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait for the industry element to be present
        try:
            industry_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="fs_icb_block"]/div/div[2]/div[1]/table/tbody/tr[1]/td[2]/strong'))
            )
            industry = industry_element.text
            # Remove the numeric code
            industry = industry.split(", ", 1)[1] if ", " in industry else industry
        except Exception as e:
            industry = None

        # Wait for the sector element to be present
        try:
            sector_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="fs_icb_block"]/div/div[2]/div[1]/table/tbody/tr[3]/td[2]/strong'))
            )
            sector = sector_element.text
            # Remove the numeric code
            sector = sector.split(", ", 1)[1] if ", " in sector else sector
        except Exception as e:
            sector = None

        # Append the data to the results DataFrame using pd.concat
        new_row = pd.DataFrame([{'ISIN': isin, 'Name': name, 'Symbol': symbol, 'Industry': industry, 'Sector': sector}])
        results = pd.concat([results, new_row], ignore_index=True)
        print('Completed: ', name)

    # Save the results to a CSV file
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_company_dir = os.path.join(project_root, 'downloads', 'processed', 'company')

    # Specify the full file path including the file name
    file_name = f"company_data.csv"
    file_path = os.path.join(processed_company_dir, file_name)

    # Save the results to a CSV file
    results.to_csv(file_path, index=False)

    # Close the browser
    driver.quit()

# Replace the existing code execution with a main function call
if __name__ == "__main__":
    main()