"""
================================================================================
                    Automated Report Generation Pipeline
        Generate comprehensive PDF reports from system metrics
================================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from reports.pdf_generator import WaterQualityPDFReport
from utils.logger import get_logger
from datetime import datetime

logger = get_logger()


def generate_report(output_path=None, title=None, author=None, organization=None):
    """
    Generate comprehensive water quality report
    
    Args:
        output_path: Custom output path (optional)
        title: Report title (optional)
        author: Report author (optional)
        organization: Organization name (optional)
    
    Returns:
        Path to generated PDF
    """
    logger.section("GENERATING PDF REPORT")
    
    # Set defaults
    if title is None:
        title = f"Water Quality Monitoring Report - {datetime.now().strftime('%B %Y')}"
    
    if author is None:
        author = "Water Quality Monitoring System"
    
    if organization is None:
        organization = "M.Tech Project - IEEE 2026"
    
    # Initialize report
    report = WaterQualityPDFReport(output_path=output_path, title=title)
    
    logger.info(f"  Title: {title}")
    logger.info(f"  Author: {author}")
    
    # Find metrics file
    try:
        from config import BASE_DIR
        metrics_path = BASE_DIR / 'outputs' / 'metrics' / 'evaluation_summary.json'
    except:
        metrics_path = Path('outputs/metrics/evaluation_summary.json')
    
    if metrics_path.exists():
        logger.info(f"  Using metrics: {metrics_path.name}")
    else:
        logger.warning("  No metrics file found - generating basic report")
        metrics_path = None
    
    # Generate report
    logger.info("\n  Generating report...")
    
    try:
        output_file = report.generate(metrics_path=str(metrics_path) if metrics_path else None)
        
        logger.success(f"\n✓ Report generated successfully!")
        logger.info(f"  Saved to: {output_file}")
        logger.info(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
        
        return output_file
    
    except Exception as e:
        logger.error(f"\n✗ Report generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    generate_report()
