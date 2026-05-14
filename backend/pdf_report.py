from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


def generate_pdf(result, output_path='reports/report.pdf'):

    c = canvas.Canvas(output_path, pagesize=letter)

    c.drawString(100, 750, 'AI Analytics Platform Report')

    y = 700

    for key, value in result.items():

        c.drawString(100, y, f'{key}: {value}')

        y -= 30

    c.save()

    return output_path