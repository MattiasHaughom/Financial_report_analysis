import logging
import schedule
import time
from app.processing.get_company_data import main as get_company_data
from app.processing.update_company_data import main as update_company_data
from app.processing.update_sector_metrics import main as update_sector_metrics
from app.processing.report_scraping import main as report_scraping
from app.processing.process_reports import main as process_reports
from app.ai_analysis import main as ai_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_weekly_tasks():
    """Run tasks that need to be executed weekly."""
    logging.info("Running weekly tasks...")
    get_company_data()
    update_company_data()
    update_sector_metrics()
    logging.info("Weekly tasks completed.")

def run_daily_tasks():
    """Run tasks that need to be executed daily."""
    logging.info("Running daily tasks...")
    new_reports = report_scraping()
    if new_reports:
        logging.info("New reports found. Processing reports...")
        process_reports()
        run_ai_analysis()
    else:
        logging.info("No new reports found. Skipping processing and AI analysis.")

def run_ai_analysis():
    """Run AI analysis and send emails."""
    logging.info("Running AI analysis...")
    ai_analysis()
    logging.info("AI analysis completed.")

def main():
    # Schedule weekly tasks to run every Monday at 10 AM CET
    schedule.every().monday.at("10:00").do(run_weekly_tasks)

    # Schedule daily tasks to run every day at 10 AM CET
    schedule.every().day.at("10:00").do(run_daily_tasks)

    logging.info("Scheduler started. Waiting for tasks to run...")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Wait for one minute

if __name__ == "__main__":
    main()