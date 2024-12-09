import requests
import os
import logging
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(filename='report_scraper.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def delete_old_files(directory, days=14):
    """Delete files older than a specified number of days in the given directory."""
    now = time.time()
    cutoff = now - (days * 86400)  # 86400 seconds in a day

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            if file_creation_time < cutoff:
                os.remove(file_path)
                print(f"Deleted old file: {filename}")

def get_today_date():
    """Get today's date in the required format."""
    return datetime.now().strftime('%Y-%m-%d')

def fetch_message_ids(api_url, params):
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
            print(f"Error: Received status code {response.status_code}")
            return []

        # Check the content type
        content_type = response.headers.get('Content-Type')
        if 'application/json' not in content_type:
            print(f"Error: Expected JSON response, got {content_type}")
            print("Response content:", response.text)
            return []

        # Parse JSON
        data = response.json()
        
        # Extract and filter report details
        reports = []
        for message in data['data']['messages']:
            # Check if the message belongs to the desired categories
            if any(category['id'] in [1001, 1002] for category in message['category']) and 'XOSL' in message['markets']:
                report_details = {
                    'messageId': message['messageId'],
                    'title': message['title'],
                    'issuerSign': message['issuerSign'],
                    'numbAttachments': message['numbAttachments'],
                    'publishedTime': message['publishedTime']
                }
                reports.append(report_details)

        return reports
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
        return []


def fetch_report_details(api_url, message_id):
    """Fetch report details for a given message ID."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Origin': 'https://newsweb.oslobors.no',
        'Referer': 'https://newsweb.oslobors.no/'
    }
    
    try:
        response = requests.post(f"{api_url}?messageId={message_id}", headers=headers)
        
        # Check if the response is successful
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} for message ID {message_id}")
            return None

        # Parse JSON
        data = response.json()
        
        # Extract message details
        message = data['data']['message']
        return message
    except requests.RequestException as e:
        print(f"Error fetching data for message ID {message_id}: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON for message ID {message_id}: {e}")
        return None

def download_attachments(message):
    """Download relevant attachments for a given message."""
    if message['numbAttachments'] > 0:
        for attachment in message['attachments']:
            attachment_id = attachment['id']
            attachment_name = attachment['name']
            
            # Check if the attachment name indicates it's a financial report
            if "report" in attachment_name.lower() or "results" in attachment_name.lower() or "account" in attachment_name.lower() or "rapport" in attachment_name.lower():
                download_url = f"https://api3.oslo.oslobors.no/v1/newsreader/attachment?messageId={message['messageId']}&attachmentId={attachment_id}"
                
                try:
                    response = requests.get(download_url)
                    response.raise_for_status()
                    
                    # Save the attachment
                    os.makedirs('downloads', exist_ok=True)
                    file_path = os.path.join('downloads', attachment_name)
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                    print(f"Downloaded: {attachment_name}")
                except requests.RequestException as e:
                    print(f"Error downloading attachment {attachment_name}: {e}")


def main():
    """Main function to execute the script."""
    list_api_url = "https://api3.oslo.oslobors.no/v1/newsreader/list"
    message_api_url = "https://api3.oslo.oslobors.no/v1/newsreader/message"
    
    # Fetch message IDs
    params = {}  # No need for specific params as filtering is done post-fetch
    reports = fetch_message_ids(list_api_url, params)
    
    # Process each report
    for report in reports:
        message_id = report['messageId']
        message = fetch_report_details(message_api_url, message_id)
        if message:
            download_attachments(message)
    
    # Clean up old files
    delete_old_files('downloads')

if __name__ == "__main__":
    main()