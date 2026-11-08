import json, glob, sys, os
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
DS, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
def F(s):
    try: return ImageFont.truetype("arial.ttf", s)
    except OSError: return ImageFont.load_default()
def log(n):
    b = json.load(open(f"{DS}/bbox_annotation/{n}.json"))["bounding_boxes"]
    return Image.open(glob.glob(f"{DS}/images/{glob.escape(n)}.tif*")[0]), b
def fit(im, w=None, h=None):
    im = im.convert("RGB"); s = min((w or 1e9)/im.width, (h or 1e9)/im.height); return im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.LANCZOS)
def panel(parts, path, H=300):
    # parts: list of (caption, image); laid out left to right, captions under each
    ims = [(c, fit(i, h=H) if i.height/i.width > 0.35 else fit(i, w=520)) for c, i in parts]
    ws = [max(i.width, 8*len(c)) for c, i in ims]; W = sum(ws) + 24*(len(ims)+1); HH = max(i.height for _, i in ims) + 56
    P = Image.new("RGB", (W, HH), "white"); d = ImageDraw.Draw(P); x = 24
    for c, i in ims:
        P.paste(i, (x, 12)); d.rectangle([x-1, 11, x+i.width, 12+i.height], outline=(200,200,200))
        d.text((x, 20+i.height), c, fill=(40,40,40), font=F(15)); x += max(i.width, 8*len(c)) + 24
    P.save(path, quality=80); print(path, P.size)
D06 = "0000000009_d-582_0000020435-0001_log06_fe_5in_md"
im, b = log(D06); L = b["lithology_legend"]; c = b["interpreted_lithology_column"]
lw, lh = L[2]-L[0], L[3]-L[1]
clay = im.crop((L[0]+int(lw*0.27), L[1]+int(lh*0.27), L[0]+int(lw*0.47), L[1]+int(lh*0.47)))
shale = im.crop((L[0]+int(lw*0.51), L[1]+int(lh*0.27), L[0]+int(lw*0.71), L[1]+int(lh*0.47)))
col = im.crop((c[0], 4609, c[2], 5209)).resize((246*2, 1200))   # shale -> siltstone boundary at 4809
panel([("Legend: Clay", clay), ("Legend: Shale", shale), ("Column: shale above, siltstone below", col)], f"{OUT}/h1_pattern_only.jpg", H=260)
im, b = log("60811405800000_608114058000_g16786_001_00_00_md5_mud_show_2"); L = b["lithology_legend"]
panel([("One legend box: lithology, modifiers, show and drilling symbols", im.crop(L))], f"{OUT}/h2_mixed_legend.jpg")
im, b = log("60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872"); L = b["lithology_legend"]
lw, lh = L[2]-L[0], L[3]-L[1]
panel([("Four cyan limestones and nine yellow sands differ only by black pattern", im.crop((L[0], L[1]+int(lh*0.12), L[0]+lw, L[1]+int(lh*0.62))))], f"{OUT}/h3_same_colour.jpg", H=320)
im, b = log("Rio_Nueces_3744_Unit_1_212_MUD_5IN"); L = b["lithology_legend"]; lw = L[2]-L[0]
full = im.crop(L); zoom = im.crop((L[0]+int(lw*0.02), L[1]+int((L[3]-L[1])*0.35), L[0]+int(lw*0.28), L[3]))
panel([("Whole legend at page scale", full), ("Same legend, 1:1 pixels", zoom.resize((zoom.width*2, zoom.height*2), Image.NEAREST))], f"{OUT}/h4_tiny_text.jpg", H=240)
im, b = log("60817411420000_mc_540_#1_st00_bp00_masterlog_3400-24195_5in_md"); c = b["interpreted_lithology_column"]
y = c[1] + int((c[3]-c[1])*0.4); blank = im.crop((c[0]-30, y, c[2]+30, y+6000)); blank = blank.resize((blank.width*4, blank.height))
im2, b2 = log("60811405800100_mdn-1_st00bp01_5 inch tvdmudlog_4560ft-22998ft"); c2 = b2["interpreted_lithology_column"]
y2 = c2[1] + int((c2[3]-c2[1])*0.4); red = im2.crop((c2[0]-30, y2, c2[2]+30, y2+6000)); red = red.resize((red.width*4, red.height))
panel([("mc_540: 6,000 rows, blank until the fill starts", blank), ("mdn-1: 6,000 rows, all one rock", red)], f"{OUT}/h5_blank_runs.jpg", H=420)
im, b = log("60812400810000_ocs-g 16987 003 st00bp00_1inmd_mudlog_8921_15721"); c = b["interpreted_lithology_column"]
y = c[1] + int((c[3]-c[1])*0.4); o1 = im.crop((c[0]-160, y+1500, c[2]+20, y+2600))
im2, b2 = log("60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872"); c2 = b2["interpreted_lithology_column"]
o2 = im2.crop((c2[0]-700, 1950, c2[2]+20, 2300))
panel([("Gap with notes inside the column", o1), ("A drilling note where the fill starts", o2)], f"{OUT}/h6_overprints.jpg", H=380)
