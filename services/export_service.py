"""
Export Service for generating PDF, Excel, and CSV reports
"""

import os
import io
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import json

logger = logging.getLogger(__name__)

class ExportService:
    def __init__(self):
        self.max_records = int(os.getenv('MAX_EXPORT_RECORDS', 1000))
        self.supported_formats = os.getenv('EXPORT_FORMATS', 'pdf,excel,csv').split(',')

    def export_predictions_to_pdf(self, predictions: List[Dict], user_info: Dict = None) -> bytes:
        """Export predictions to PDF format"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6
            )
            
            # Build content
            content = []
            
            # Title
            content.append(Paragraph("Health Prediction Report", title_style))
            content.append(Spacer(1, 20))
            
            # User info
            if user_info:
                content.append(Paragraph("User Information", heading_style))
                user_table_data = [
                    ["Name:", user_info.get('name', 'N/A')],
                    ["Email:", user_info.get('email', 'N/A')],
                    ["Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                ]
                user_table = Table(user_table_data, colWidths=[1.5*inch, 4*inch])
                user_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('BACKGROUND', (1, 0), (1, -1), colors.beige),
                ]))
                content.append(user_table)
                content.append(Spacer(1, 20))
            
            # Predictions
            content.append(Paragraph("Prediction Results", heading_style))
            
            if predictions:
                # Create table data
                table_data = [["Date", "Type", "Prediction", "Confidence", "Risk Level"]]
                
                for pred in predictions:
                    table_data.append([
                        pred.get('timestamp', 'N/A'),
                        pred.get('type', 'N/A'),
                        pred.get('prediction', 'N/A'),
                        f"{pred.get('confidence', 0):.1f}%",
                        pred.get('risk_level', 'N/A')
                    ])
                
                # Create table
                table = Table(table_data, colWidths=[1.2*inch, 1.5*inch, 1.5*inch, 1*inch, 1*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                content.append(table)
            else:
                content.append(Paragraph("No predictions found.", normal_style))
            
            content.append(Spacer(1, 20))
            
            # Summary
            content.append(Paragraph("Summary", heading_style))
            total_predictions = len(predictions)
            high_risk_count = len([p for p in predictions if p.get('risk_level') == 'High'])
            
            summary_text = f"""
            Total Predictions: {total_predictions}<br/>
            High Risk Predictions: {high_risk_count}<br/>
            Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
            <br/>
            <b>Disclaimer:</b> This report is for informational purposes only and should not replace professional medical advice. 
            Please consult with healthcare professionals for proper medical evaluation and treatment.
            """
            
            content.append(Paragraph(summary_text, normal_style))
            
            # Build PDF
            doc.build(content)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            raise

    def export_predictions_to_excel(self, predictions: List[Dict], user_info: Dict = None) -> bytes:
        """Export predictions to Excel format"""
        try:
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Main predictions sheet
                if predictions:
                    df = pd.DataFrame(predictions)
                    df.to_excel(writer, sheet_name='Predictions', index=False)
                else:
                    # Create empty DataFrame with proper columns
                    df = pd.DataFrame(columns=['timestamp', 'type', 'prediction', 'confidence', 'risk_level'])
                    df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Predictions', 'High Risk Count', 'Average Confidence', 'Report Date'],
                    'Value': [
                        len(predictions),
                        len([p for p in predictions if p.get('risk_level') == 'High']),
                        sum(p.get('confidence', 0) for p in predictions) / len(predictions) if predictions else 0,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # User info sheet
                if user_info:
                    user_data = {
                        'Field': ['Name', 'Email', 'Report Generated'],
                        'Value': [
                            user_info.get('name', 'N/A'),
                            user_info.get('email', 'N/A'),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ]
                    }
                    user_df = pd.DataFrame(user_data)
                    user_df.to_excel(writer, sheet_name='User Info', index=False)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating Excel: {str(e)}")
            raise

    def export_predictions_to_csv(self, predictions: List[Dict], user_info: Dict = None) -> str:
        """Export predictions to CSV format"""
        try:
            if predictions:
                df = pd.DataFrame(predictions)
                csv_content = df.to_csv(index=False)
            else:
                # Create empty CSV with headers
                csv_content = "timestamp,type,prediction,confidence,risk_level\n"
            
            return csv_content
            
        except Exception as e:
            logger.error(f"Error generating CSV: {str(e)}")
            raise

    def get_export_filename(self, format_type: str, user_id: str = None) -> str:
        """Generate filename for export"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_suffix = f"_{user_id}" if user_id else ""
        return f"health_predictions{user_suffix}_{timestamp}.{format_type}"

    def validate_export_request(self, format_type: str, record_count: int) -> Dict[str, Any]:
        """Validate export request"""
        if format_type not in self.supported_formats:
            return {
                'valid': False,
                'error': f'Unsupported format. Supported formats: {", ".join(self.supported_formats)}'
            }
        
        if record_count > self.max_records:
            return {
                'valid': False,
                'error': f'Too many records. Maximum allowed: {self.max_records}'
            }
        
        return {'valid': True, 'error': None}

# Global export service instance
export_service = ExportService()
