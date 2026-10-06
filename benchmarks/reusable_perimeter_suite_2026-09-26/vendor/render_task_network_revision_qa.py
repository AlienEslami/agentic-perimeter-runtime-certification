"""Poppler page renders and contact sheets for manual manuscript layout QA."""
import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw
from pypdf import PdfReader


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report={}
    terms=['Positivity, queue comparison','Uncertainty range and the cost',
           'Live requests on the primary','Structured certificates during',
           'Independent positivity and queue','Signed propagation and robust',
           'Validated matrix functions','Binding-service online comparison',
           'Online missions. Contract','Network form of the class',
           'Connected three-region','Typed task operations','Task-State Interface',
           'Computational dependence','Primary-model task-state','Memory corruption',
           'held-out','Operation &','Temporal consistency']
    for short,stem in [('journal','FOUNDATION_MFD_PARTB_ELSEVIER'),('ieee','FOUNDATION_MFD_PARTB_IEEE_READING_COPY')]:
        source=args.root/(stem+'.pdf')
        pages=PdfReader(source).pages
        directory=args.output/short
        directory.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-r','65','-png',str(source),str(directory/'page')],check=True)
        rasters=sorted(directory.glob('page-*.png'))
        assert len(rasters)==len(pages)
        contacts=[]
        for begin in range(0,len(rasters),16):
            sheet=Image.new('RGB',(1000,1420),'#e5e5e5')
            draw=ImageDraw.Draw(sheet)
            for j,path in enumerate(rasters[begin:begin+16]):
                with Image.open(path) as im:
                    im.thumbnail((238,326))
                    left=250*(j%4)+(250-im.width)//2
                    top=355*(j//4)+23
                    sheet.paste(im,(left,top))
                draw.text((250*(j%4)+9,355*(j//4)+5),f'{short} page {begin+j+1}',fill='black')
            target=directory/f'contact-{begin//16+1}.png'
            sheet.save(target);contacts.append(str(target))
        hits={}
        selected={1}
        for number,page in enumerate(pages,1):
            text=' '.join((page.extract_text() or '').split())
            matched=[term for term in terms if term.lower() in text.lower()]
            if matched:
                hits[number]=matched
                selected.add(number)
                if number<len(pages): selected.add(number+1)
        details=[]
        for number in sorted(selected):
            target=directory/f'detail-{number:02d}'
            subprocess.run(['pdftoppm','-f',str(number),'-l',str(number),'-r','140','-singlefile','-png',str(source),str(target)],check=True)
            details.append(str(target.with_suffix('.png')))
        report[short]=dict(pages=len(pages),contacts=contacts,detail_pages=sorted(selected),hits=hits,details=details)
    (args.output/'render_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
