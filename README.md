# Financial Report Processing Script

## Overall Application Structure

### report_scraping.py
* Purpose: Download reports from NewsWeb for Norwegian companies.
* Logic:
    * Connect to the NewsWeb API to fetch the latest reports.
    * Filter reports based on desired criteria (e.g., category, market).
    * Download report attachments (PDFs) and save metadata.
    * Delete old files to manage storage.

### process_reports.py
* Purpose: Process the downloaded reports to extract key information.
* Logic:
    * Load report metadata and process each report.
    * Extract relevant pages from PDFs (e.g., using keyword searches).
    * Convert PDFs to text using PDF parsers.
    * Chunk the extracted text for embedding.
    * Generate embeddings for each text chunk.
    * Upsert text, embeddings, and metadata into the reports table in your database.

### ai_analysis.py
* Purpose: Analyze processed reports using AI models guided by sector-specific metrics.
* Logic:
    * Fetch new reports from the reports table that need analysis.
    * For each report:
        * Retrieve company information from the company table, including sector.
        * Retrieve sector-specific metrics from the sector_metrics table.
        * Construct a detailed prompt incorporating sector metrics and company data.
        * Use pydantic_ai to define structured prompts and expected response models.
        * Generate AI analysis using the model and save the analysis, along with its embedding, into the analysis table.

   
### Features
- Fetch Message IDs: Retrieves message IDs from the API based on specific categories and market criteria.
- Download Attachments: Downloads attachments from the messages that are identified as financial reports.
- File Management: Deletes files older than a specified number of days to manage storage space.
- Logging: Logs the operations and any errors encountered during execution.


## Requirements
- Python 3.x
- Libraries: requests, pdfplumber, PyPDF2

## Install the required libraries using pdm:
```
pdm init
```


## Future Enhancements
### AI Analysis Integration
- Data Extraction: Implement AI models to analyze the extracted data and generate summaries.
- Storage: Save the AI-generated summaries in a database or file system for future retrieval and analysis.
- User Interface: Develop a user interface to visualize the analysis results and allow users to interact with the data.
### Additional Features
- Batch Processing: Enhance the script to process large volumes of reports more efficiently.
- Parallel Processing: Utilize parallel processing to speed up the extraction and analysis of reports.
- Advanced Logging: Implement more detailed logging to track the performance and outcomes of the AI analysis.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License.