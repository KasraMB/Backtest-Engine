import re, html, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
for f in sorted(glob.glob('art_*.html')):
    print(f"=========== {f} ===========")
    t = open(f, encoding='utf-8').read()
    m = re.search(r'<article[^>]*>(.*?)</article>', t, flags=re.S)
    t = m.group(1) if m else t
    t = re.sub(r'<script.*?</script>', '', t, flags=re.S)
    t = re.sub(r'<style.*?</style>', '', t, flags=re.S)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = html.unescape(t)
    t = re.sub(r'\s+', ' ', t)
    print(t[:7000])
    print()
