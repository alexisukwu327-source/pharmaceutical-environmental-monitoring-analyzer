# Pharmaceutical Environmental Monitoring Analyzer

A Python-based environmental monitoring analytics application designed to help visualize microbial monitoring results, identify potential quality concerns, and automate routine data reporting.

This project combines my experience in pharmaceutical manufacturing environments with my interest in software engineering, data analytics, and building practical technology solutions for regulated industries.

## Project Overview

Environmental monitoring is an important part of maintaining controlled pharmaceutical manufacturing environments. Reviewing monitoring data, identifying results that exceed established limits, and documenting findings are essential quality activities.

I developed this application to explore how Python-based tools can simplify these workflows and make monitoring results easier to interpret.

**This is an educational portfolio project using synthetic data. It is not a validated pharmaceutical quality system.**

## Features

- **CSV Data Upload:** Import environmental monitoring datasets for analysis.
- **Data Validation:** Identify and exclude records with missing or invalid required values.
- **Monitoring Dashboard:** Display total samples, average microbial counts, and results exceeding configured limits.
- **Trend Visualization:** Plot microbial counts over time against example monitoring limits.
- **Automated Quality Reporting:** Generate downloadable summaries highlighting results requiring review.
- **Quality Investigation Assistant:** Answer predefined questions about monitoring trends, exceedances, and potential areas for investigation.
- **AI Quality Insights (Optional):** Integrates with the OpenAI API to generate natural-language summaries grounded in uploaded synthetic monitoring data. Requires a separately configured API key, billing, and available quota.
- **CSV Export:** Download analyzed monitoring results.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Streamlit
- OpenAI API (optional integration)
- CSV data processing

## How to Run the Project

1. Clone or download this repository.
2. Open the project folder in a terminal.
3. Create and activate a Python virtual environment.
4. Install the required packages:

   `python -m pip install -r requirements.txt`

5. Start the application:

   `python -m streamlit run app.py`

6. Open the local Streamlit URL displayed in your terminal.
7. Upload `sample_data.csv` to explore the dashboard.

To enable the optional AI assistant, create `.streamlit/secrets.toml` containing an `OPENAI_API_KEY` entry. Keep this file private and never commit API credentials.

## Example Analysis

The included synthetic dataset demonstrates how the application identifies monitoring results exceeding configured example limits.

For example, a sample from Room A on October 4, 2026, records 12 CFU against an example limit of 10 CFU. The application flags this result, displays the trend, and includes it in the automated quality report.

The flagged result indicates a need for further review; it does not establish the cause of the exceedance.

## Project Motivation

Through my experience in pharmaceutical environmental monitoring and biomedical engineering research, I became interested in how software could support technical teams working with complex data.

I wanted to build something that connected my understanding of regulated manufacturing environments with my computer science background.

This project demonstrates how domain knowledge and software development can work together to solve practical data-analysis problems.

## Future Improvements

- Add automated tests for data validation and exceedance detection.
- Support configurable monitoring limits and additional sampling locations.
- Improve AI response evaluation and data-grounding checks.
- Expand reporting and visualization capabilities.
- Explore secure deployment and containerization.

## Important Disclaimer

All sample data is synthetic. This application is for educational demonstration only and must not be used for regulated quality decisions, product release, or patient-safety decisions.
