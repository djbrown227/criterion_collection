# export_findings.py — builds findings.json for about.html from closet_data/. Same logic as the notebook.
import json, re, itertools, collections
import pandas as pd, numpy as np
from pathlib import Path

DATA = Path('closet_data'); OUT = Path('findings.json')
col  = pd.read_csv(DATA / 'collection.csv'); det = pd.read_csv(DATA / 'collection_details.csv')
pk   = pd.read_csv(DATA / 'collection_picks.csv'); vis = pd.read_csv(DATA / 'collections.csv')
cast = pd.read_csv(DATA / 'cast.csv').dropna(subset=['spine']); cred = pd.read_csv(DATA / 'credits.csv').dropna(subset=['spine'])
bs   = pd.read_csv(DATA / 'boxset_films.csv')

col = col.dropna(subset=['spine']).copy(); col['spine'] = col.spine.astype(int); col = col.drop_duplicates('spine').sort_values('spine')
det = det.dropna(subset=['spine']).copy(); det['spine'] = det.spine.astype(int); det = det.drop_duplicates('spine').set_index('spine')
col['year'] = col.year.fillna(col.spine.map(det.year)); col['country'] = col.country.fillna(col.spine.map(det.country))
col['c1'] = col.country.fillna('').str.split(',').str[0].str.strip(); col['dec'] = (col.year // 10 * 10)
col['runtime'] = col.spine.map(det.runtime_min); col['bw'] = col.spine.map(det.color).fillna('').str.lower().str.contains('black')
col['fid'] = col.url.str.extract(r'/films/(\d+)-')[0]
col['in_set'] = col.fid.isin(set(bs.film_url.str.extract(r'/films/(\d+)-')[0].dropna()))
cast['spine'] = cast.spine.astype(int); cred['spine'] = cred.spine.astype(int)
vis['guest'] = vis.page_title.str.replace(r"(?:['’]s|['’])\s+(?:Mobile\s+)?Closet Picks.*$", '', regex=True).str.strip('​ ').str.strip()
vis['date'] = pd.to_datetime(vis.date_recorded)
p = pk.dropna(subset=['spine']).copy(); p['spine'] = p.spine.astype(int)
p = p.drop(columns=['guest']).merge(vis[['collection_id', 'guest', 'date']], on='collection_id')
p = p.merge(col[['spine', 'year', 'c1', 'dec', 'director']].rename(columns={'year': 'fyear', 'director': 'dir'}), on='spine', how='left')
p = p.sort_values(['date', 'collection_id', 'pick_order']).reset_index(drop=True)
N = p.collection_id.nunique(); cnt = p.groupby('spine').collection_id.nunique(); col['n'] = col.spine.map(cnt).fillna(0).astype(int)
p['pop'] = p.spine.map(cnt)

# shelf proxy
seen, shelf = [], {}
for cid, g in p.groupby('collection_id', sort=False):
    seen += list(g.spine); s = sorted(seen, reverse=True); shelf[cid] = s[2] if len(s) >= 3 else s[-1]
shelf = pd.Series(shelf); p['shelf'] = p.collection_id.map(shelf)
ahead = int((p.spine > p.shelf + 15).sum())

standalone = col[(~col.in_set) & (~col.is_set)]
head = {'visits': int(N), 'picks': int(len(p)), 'distinct': int(p.spine.nunique()), 'spines': int(len(col)), 'once': int((cnt == 1).sum()), 'ten_plus': int((cnt >= 10).sum()),
        'top20_share': round(float(cnt.sort_values(ascending=False).head(20).sum() / len(p) * 100), 1), 'standalone': int(len(standalone)), 'never_standalone': int((standalone.n == 0).sum()),
        'first_date': vis.date.min().strftime('%B %Y'), 'last_date': vis.date.max().strftime('%B %Y'), 'median_per_visit': float(p.groupby('collection_id').size().median()), 'ahead_of_shelf': ahead,
        'pairs_sharing_2': None}
sets = p.groupby('collection_id').apply(lambda g: set(g.spine))
head['pairs_sharing_2'] = int(sum(1 for i, j in itertools.combinations(sets.values, 2) if len(i & j) >= 2)); head['pairs_total'] = int(len(sets) * (len(sets) - 1) // 2)

strip = [{'s': int(r.spine), 't': r.title, 'n': int(r.n), 'y': int(r.year) if pd.notna(r.year) else None, 'set': bool(r.is_set), 'inset': bool(r.in_set)} for r in col.itertuples()]
top = col.sort_values('n', ascending=False).head(20)[['spine', 'title', 'director', 'year', 'n']].values.tolist()

# grid
films = col.dropna(subset=['dec'])[['spine', 'c1', 'dec']]; cache = {}
def shares(sh):
    if sh not in cache:
        f = films[films.spine <= sh]; cache[sh] = (f.groupby(['c1', 'dec']).size() / len(f), f.c1.value_counts(normalize=True), f.dec.value_counts(normalize=True))
    return cache[sh]
exp_cell, exp_c, exp_d = collections.Counter(), collections.Counter(), collections.Counter()
for cid, g in p.groupby('collection_id'):
    cs, cc, cd = shares(shelf[cid]); n = len(g)
    for k, v in cs.items(): exp_cell[k] += v * n
    for k, v in cc.items(): exp_c[k] += v * n
    for k, v in cd.items(): exp_d[k] += v * n
obs = p.groupby(['c1', 'dec']).size()
cells = [[k[0], int(k[1]), int(obs.get(k, 0)), round(e, 1), round(obs.get(k, 0) / e, 2)] for k, e in exp_cell.items() if e >= 8]
countries = ['United States', 'France', 'Japan', 'Italy', 'United Kingdom', 'Germany', 'Sweden', 'Soviet Union', 'Mexico', 'Canada', 'Poland', 'Hong Kong']
cg = films.groupby(['c1', 'dec']).size()
coll_grid = {c: {int(d): int(cg.get((c, d), 0)) for d in range(1920, 2030, 10)} for c in countries}
grid = {'cells': cells, 'country': {c: [int(p.c1.value_counts().get(c, 0)), round(exp_c[c], 1)] for c in exp_c if exp_c[c] >= 15},
        'decade': {int(d): [int((p.dec == d).sum()), round(exp_d[d], 1)] for d in exp_d if exp_d[d] >= 20}}

# reappraisal
c = col.dropna(subset=['year']).copy()
c['tercile'] = c.groupby('dec').spine.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=['early', 'mid', 'late']))
c['exposed'] = [(shelf >= s).sum() for s in c.spine]; c = c[c.exposed >= 20]; c['rate'] = c.n / c.exposed * 100
reap = c.pivot_table(index='dec', columns='tercile', values='rate', aggfunc='mean', observed=True).loc[1920:2000]
reap = [[int(d), round(r['early'], 2), round(r['mid'], 2), round(r['late'], 2)] for d, r in reap.iterrows()]
late = c[(c.year < 1995) & (c.spine > 900)].sort_values('n', ascending=False).head(10)[['spine', 'title', 'year', 'n']].values.tolist()
overall = {t: round(float(c[c.tercile == t].rate.mean()), 2) for t in ['early', 'mid', 'late']}

# directors
dd = p.dropna(subset=['dir']).groupby('dir').agg(picks=('collection_id', 'size'), titles=('spine', 'nunique')).reset_index()
directors = dd[dd.picks >= 10].sort_values('picks', ascending=False).values.tolist()

# pick order
po = p[p.pick_order <= 8].groupby('pick_order').agg(n=('spine', 'size'), mean_pop=('pop', 'mean'), mean_year=('fyear', 'mean')).round(2).reset_index()
pickorder = po[['pick_order', 'mean_pop', 'mean_year', 'n']].values.tolist()

# adjacent spines
def adj(s, d=3): s = sorted(s); return sum(1 for a, b in zip(s, s[1:]) if b - a <= d)
obs_adj = int(sum(adj(s) for s in sets.values)); rng = np.random.default_rng(1); sims = []
for _ in range(200):
    tot = 0
    for cid, s in sets.items(): tot += adj(rng.choice(films[films.spine <= shelf[cid]].spine.values, len(s), replace=False))
    sims.append(tot)
neighbors = {'observed': obs_adj, 'random_mean': round(float(np.mean(sims)), 1), 'random_sd': round(float(np.std(sims)), 1)}

# vanity
def norm(s): return re.sub(r'[^a-z ]', '', str(s).lower()).strip()
people = {}
for sp, g in cast.groupby('spine'): people.setdefault(sp, set()).update(norm(x) for x in g.actor)
for sp, g in cred.groupby('spine'): people.setdefault(sp, set()).update(norm(x) for x in g.name)
van = []
for r in p.itertuples():
    names = [norm(x) for x in re.split(r' and |&|,', r.guest)]; ppl = people.get(r.spine, set())
    if any(nm and any(nm == q or (len(nm.split()) >= 2 and nm in q) for q in ppl) for nm in names): van.append({'guest': r.guest, 'title': r.title, 'year': int(r.date.year)})

# fair leaderboard
expo = pd.Series({s: int((shelf >= s).sum()) for s in col.spine})
lb = col.set_index('spine').join(expo.rename('exposed')); lb = lb[(lb.exposed >= 60) & (~lb.in_set)]; lb['rate'] = lb.n / lb.exposed * 100
raw50 = set(lb.sort_values('n', ascending=False).head(50).index)
fair = [[int(s), r.title, r.director, int(r.year) if pd.notna(r.year) else None, int(r.n), int(r.exposed), round(r.rate, 2), s not in raw50] for s, r in lb.sort_values('rate', ascending=False).head(20).iterrows()]

# drift
yr = p.groupby(p.date.dt.year)
drift = []
for y, g in yr:
    f = films[films.spine <= g.shelf.max()]
    drift.append([int(y), int(len(g)), round(float((g.c1 == 'United States').mean()), 3), round(float((f.c1 == 'United States').mean()), 3), round(float(g.fyear.median()), 1), round(float(col[col.spine <= g.shelf.max()].year.median()), 1)])

# people
def people_table(role_df, name_col, min_films=3, order_max=None):
    d = role_df if order_max is None else role_df[role_df.order <= order_max]
    d = d[['spine', name_col]].drop_duplicates().rename(columns={name_col: 'name'})
    onshelf = d.groupby('name').spine.nunique().rename('releases')
    taken = d.merge(col[col.n > 0][['spine']], on='spine').groupby('name').spine.nunique().rename('taken')
    visits = d.merge(p[['spine', 'collection_id']], on='spine').groupby('name').collection_id.nunique().rename('visits')
    t = pd.concat([onshelf, taken, visits], axis=1).fillna(0).astype(int); t['rate'] = (t.visits / t.releases).round(2)
    return t[t.releases >= min_films]
def role(keys): return cred[cred.credit.str.lower().isin(keys)]
DP = ['cinematography', 'director of photography', 'cinematographer', 'photography']
act = people_table(cast, 'actor', 3, order_max=6)
dp = people_table(role(DP), 'name', 3)
dd_ = role(['director', 'directed by'])[['spine', 'name']].rename(columns={'name': 'director'}); dp_ = role(DP)[['spine', 'name']].rename(columns={'name': 'dp'})
team = dd_.merge(dp_, on='spine').drop_duplicates().merge(col[['spine', 'n']], on='spine').groupby(['director', 'dp']).agg(releases=('spine', 'nunique'), visits=('n', 'sum')).reset_index()
people_out = {
    'actors': [[i, int(r.releases), int(r.taken), int(r.visits)] for i, r in act.sort_values('visits', ascending=False).head(12).iterrows()],
    'dps': [[i, int(r.releases), int(r.taken), int(r.visits)] for i, r in dp.sort_values('visits', ascending=False).head(10).iterrows()],
    'dps_rare': [[i, int(r.releases), int(r.taken), int(r.visits)] for i, r in dp[dp.releases >= 6].sort_values('rate').head(8).iterrows()],
    'teams': [[r.director, r.dp, int(r.releases), int(r.visits)] for r in team[team.releases >= 2].sort_values('visits', ascending=False).head(12).itertuples()],
}

# runtime / bw
pp = p.merge(col[['spine', 'runtime', 'bw']], on='spine'); bins = [0, 90, 110, 130, 160, 400]; labels = ['under 90', '90–110', '110–130', '130–160', 'over 160']
runtime = {'labels': labels, 'shelf': (pd.cut(col.runtime, bins, labels=labels).value_counts(normalize=True).sort_index() * 100).round(1).tolist(),
           'picks': (pd.cut(pp.runtime, bins, labels=labels).value_counts(normalize=True).sort_index() * 100).round(1).tolist(), 'bw': [round(float(pp.bw.mean() * 100), 1), round(float(col.bw.mean() * 100), 1)]}

json.dump({'head': head, 'strip': strip, 'top': top, 'grid': grid, 'coll_grid': coll_grid, 'reap': reap, 'reap_overall': overall, 'late': late, 'directors': directors,
           'pickorder': pickorder, 'neighbors': neighbors, 'vanity': van, 'fair': fair, 'drift': drift, 'people': people_out, 'runtime': runtime}, OUT.open('w'), default=str)
print('wrote', OUT, OUT.stat().st_size // 1024, 'KB')
