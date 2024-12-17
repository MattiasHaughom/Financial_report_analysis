import requests
import os
import logging
import time
from datetime import datetime, timedelta
import json
from typing import List, Dict

# Configure logging
logging.basicConfig(
    filename='report_scraper.log',
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def delete_old_files(directory: str, days: int = 14) -> None:
    """Delete files older than a specified number of days in the given directory."""
    now = time.time()
    cutoff = now - (days * 86400)  # 86400 seconds in a day

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            if file_creation_time < cutoff:
                os.remove(file_path)
                logging.info(f"Deleted old file: {filename}")

def fetch_message_ids(api_url: str, params: dict) -> List[dict]:
    """Fetch message IDs from the API response using a POST request."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Origin': 'https://newsweb.oslobors.no',
        'Referer': 'https://newsweb.oslobors.no/',
    }

    try:
        response = requests.post(api_url, headers=headers, json=params)

        # Check if the response is successful
        if response.status_code != 200:
            logging.error(f"Error: Received status code {response.status_code}")
            return []

        # Check the content type
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            logging.error(f"Error: Expected JSON response, got {content_type}")
            logging.error(f"Response content: {response.text}")
            return []

        # Parse JSON
        data = response.json()

        # Extract and filter report details
        reports = []
        for message in data.get('data', {}).get('messages', []):
            # Check if the message belongs to the desired categories and markets
            if any(category['id'] in [1001, 1002] for category in message.get('category', [])) and 'XOSL' in message.get('markets', []):
                report_details = {
                    'messageId': message['messageId'],
                    'title': message['title'],  # Using title as the attachment name
                    'issuerSign': message['issuerSign'],
                    'numbAttachments': message['numbAttachments'],
                    'publishedTime': message['publishedTime']
                }
                reports.append(report_details)
        
        logging.info(f"Successfully fetched {len(reports)} reports")
        return reports

    except requests.RequestException as e:
        logging.error(f"RequestException during fetch_message_ids: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError during fetch_message_ids: {e}")
        return []

def fetch_report_details(message_api_url: str, message_id: int) -> Dict:
    """Fetch attachment details for a given message ID."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Origin': 'https://newsweb.oslobors.no',
        'Referer': 'https://newsweb.oslobors.no/',
    }

    try:
        response = requests.get(f"{message_api_url}?messageId={message_id}", headers=headers)
        response.raise_for_status()

        # Check the content type
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            logging.error(f"Error: Expected JSON response, got {content_type}")
            logging.error(f"Response content: {response.text}")
            return {}

        # Parse JSON
        data = response.json()

        # Extract attachment details
        message_data = data.get('data', {}).get('message', {})
        attachments = message_data.get('attachments', [])

        # Prepare attachment details
        attachment_details = [
            {
                'attachmentId': attachment['id'],
                'attachmentName': attachment['name']
            }
            for attachment in attachments
        ]
        
        return {
            'messageId': message_id,
            'attachments': attachment_details
        }

    except requests.RequestException as e:
        logging.error(f"RequestException during fetch_report_details for message ID {message_id}: {e}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError during fetch_report_details for message ID {message_id}: {e}")
        return {}

def download_attachments(report: Dict, downloads_dir: str) -> List[str]:
    """Download relevant attachments for a given report and return the filenames."""
    downloaded_files = []
    message_id = report['messageId']
    attachments = report.get('attachments', [])

    if not attachments:
        logging.warning(f"No attachments found for message ID {message_id}")
        return downloaded_files

    for attachment in attachments:
        attachment_id = attachment['attachmentId']
        attachment_name = attachment['attachmentName']

        # Construct the download URL
        download_url = f"https://api3.oslo.oslobors.no/v1/newsreader/attachment?messageId={message_id}&attachmentId={attachment_id}"

        try:
            response = requests.get(download_url)
            response.raise_for_status()

            # Save the attachment to the raw directory
            file_path = os.path.join(downloads_dir, attachment_name)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            logging.info(f"Downloaded: {attachment_name}")
            downloaded_files.append(attachment_name)
        except requests.RequestException as e:
            logging.error(f"Error downloading attachment {attachment_name} for message ID {message_id}: {e}")

    return downloaded_files

def save_reports_to_file(reports: List[dict], ticker_symbol: str, filename: str = 'reports_data.json') -> None:
    """Save the collected reports data to a JSON file with the ticker symbol in the file name."""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    metadata_path = os.path.join(project_root, 'data', 'processed', 'metadata')

    # Ensure metadata directory exists
    os.makedirs(metadata_path, exist_ok=True)

    # Save the file
    filepath = os.path.join(metadata_path, f"{ticker_symbol}_{filename}")
    with open(filepath, 'w') as file:
        json.dump(reports, file, indent=4)
    logging.info(f"Saved reports metadata to {filepath}")

def main():
    """Main function to execute the script."""
    # API URLs
    list_api_url = "https://api3.oslo.oslobors.no/v1/newsreader/list"
    message_api_url = "https://api3.oslo.oslobors.no/v1/newsreader/message"

    params = {}  # Add necessary parameters if required

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Set up directory paths
    downloads_dir = os.path.join(project_root, 'data', 'raw')       # Raw PDFs
    processed_reports_dir = os.path.join(project_root, 'data', 'processed', 'reports')  # Processed PDFs
    metadata_dir = os.path.join(project_root, 'data', 'processed', 'metadata')        # Metadata JSON files

    # Ensure directories exist
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(processed_reports_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Clean up old raw files
    delete_old_files(downloads_dir)

    # Fetch message IDs and other details
    reports = fetch_message_ids(list_api_url, params)
    logging.info(f"Fetched {len(reports)} reports from the announcements API.")

    # For each report, fetch attachment details and download attachments
    for report in reports:
        message_id = report['messageId']

        # Fetch attachment details
        report_details = fetch_report_details(message_api_url, message_id)
        if not report_details:
            continue  # Skip if fetching details failed

        # Update the report with attachment details
        report['attachments'] = report_details.get('attachments', [])

        # Download attachments
        downloaded_files = download_attachments(report, downloads_dir)
        report['downloadedFiles'] = downloaded_files  # Optionally track downloaded files

    # Save the updated reports with attachment details to metadata JSON
    if reports:
        # Assuming all reports have the same 'issuerSign'; adjust if not
        ticker_symbol = reports[0]['issuerSign']
        save_reports_to_file(reports, ticker_symbol)
        logging.info(f"Processed and saved {len(reports)} reports for ticker {ticker_symbol}.")
    else:
        logging.info("No reports fetched from the API.")

if __name__ == "__main__":
    main()