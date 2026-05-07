import pickle
with open('sweep_results_amr_v2/checkpoint_A.pkl', 'rb') as f:
    ck = pickle.load(f)
results = sorted(ck['results'], key=lambda r: -r['sortino'])
print(f'Done: {ck["done"]}  Valid: {len(results)}')
print(f'{'#':>3}  {'Sessions':<20}  {'N':>5}  {'tpd':>5}  {'WR':>6}  {'avgR':>7}  {'Sortino':>8}')
for i, r in enumerate(results[:50], 1):
    p = r['params']
    s = '+'.join(p['sessions'])
    w = '+'.join(f'{k}={v}' for k,v in p['window_mins_per_session'].items())
    print(f'{i:>3}  {s+"|"+w:<20}  {r["n_t"]:>5}  {r["tpd"]:>5.2f}  {r["wr"]:>6.1%}  {r["avgR"]:>+7.4f}  {r["sortino"]:>8.3f}')
