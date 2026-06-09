from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


A_EXT = [
    "Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt.",
    "The Industrial Revolution began in Great Britain in the late 18th century.",
    "Charles Darwin published On the Origin of Species in 1859.",
    "The first programmable digital computers were built during the 1940s.",
    "The Mariana Trench is the deepest known location in the ocean.",
    "Saturn has the most extensive ring system of any planet in the solar system.",
    "The internet was originally developed as a US military communication network.",
    "Vincent van Gogh painted The Starry Night in June 1889.",
    "The American Civil War lasted from 1861 to 1865.",
    "The Sahara is the largest hot desert in the world.",
    "Rosalind Franklin contributed crucial X-ray diffraction data to the discovery of DNA structure.",
    "The fall of Constantinople in 1453 marked the end of the Byzantine Empire.",
    "Helium is the second most abundant element in the universe after hydrogen.",
    "The Magna Carta was signed by King John of England in 1215.",
    "Honeybees communicate the location of food sources through a waggle dance.",
    "The first successful airplane flight by the Wright brothers occurred in 1903.",
    "Mount Vesuvius erupted in 79 CE, burying the Roman cities of Pompeii and Herculaneum.",
    "The Hubble Space Telescope was launched into low Earth orbit in 1990.",
    "Antibiotics were first widely used during the Second World War.",
    "The Renaissance is generally considered to have begun in 14th-century Italy.",
    "Pluto was reclassified from a planet to a dwarf planet in 2006.",
    "The Higgs boson was experimentally confirmed at CERN in 2012.",
    "The first heart transplant was performed by Christiaan Barnard in 1967.",
    "Greenland is the world's largest island.",
    "The Russian Revolution of 1917 led to the formation of the Soviet Union.",
    "Plate tectonics theory was developed in the mid-20th century.",
    "The Suez Canal connects the Mediterranean Sea to the Red Sea.",
    "Antarctica was discovered by humans in the early 19th century.",
    "The Big Bang theory describes the origin of the universe approximately 13.8 billion years ago.",
    "Augustus became the first Roman emperor in 27 BCE.",
    "The European Union was established by the Maastricht Treaty in 1993.",
    "The first successful vaccine was developed by Edward Jenner against smallpox.",
    "Mozart composed his first symphony at the age of eight.",
    "The Channel Tunnel between England and France opened in 1994.",
    "Electricity became commercially available in the late 19th century.",
    "The Tokugawa shogunate ruled Japan from 1603 to 1868.",
    "The Andes mountain range runs along the western coast of South America.",
    "The Mona Lisa is housed in the Louvre Museum in Paris.",
    "The first photograph was taken by Joseph Nicéphore Niépce around 1826.",
    "The Wall Street Crash of 1929 triggered the Great Depression.",
    "Telomeres are repetitive nucleotide sequences at the ends of chromosomes.",
    "The Battle of Waterloo in 1815 ended Napoleon's rule as Emperor of the French.",
    "Wikipedia was launched on January 15, 2001.",
    "The Galápagos Islands are home to many endemic species.",
    "CRISPR-Cas9 is a genome-editing tool derived from a bacterial immune system.",
    "The Trans-Siberian Railway is the longest railway line in the world.",
    "The Manhattan Project produced the first nuclear weapons during World War II.",
    "Michelangelo sculpted the statue of David between 1501 and 1504.",
    "Black holes were predicted by Albert Einstein's general theory of relativity.",
    "The Dead Sea is one of the saltiest bodies of water on Earth.",
    "The Library of Congress is the largest library in the world by shelf space.",
    "Bitcoin was created by an unknown person or group using the pseudonym Satoshi Nakamoto.",
    "The Khmer Rouge regime ruled Cambodia from 1975 to 1979.",
    "The Apollo 13 mission successfully returned to Earth despite an oxygen tank explosion.",
    "Photons are massless particles that mediate the electromagnetic force.",
    "The Iron Age followed the Bronze Age in many parts of Eurasia.",
    "The James Webb Space Telescope launched in December 2021.",
    "The South Pole was first reached by Roald Amundsen in December 1911.",
    "The Industrial Internet of Things connects machines and infrastructure to networks.",
    "The Holocaust resulted in the murder of approximately six million Jews during World War II.",
    "The longest-living tree on record is a Great Basin bristlecone pine over 5,000 years old.",
    "The first heart-lung machine was used in human surgery in 1953.",
    "Continental drift was first proposed by Alfred Wegener in 1912.",
    "The Great Schism of 1054 divided Christianity into Catholic and Eastern Orthodox branches.",
    "mRNA vaccines were authorized for emergency use against COVID-19 in late 2020.",
    "The Suez Crisis of 1956 marked a turning point in British and French global influence.",
    "The Pythagorean theorem relates the side lengths of a right triangle.",
    "Tides are caused primarily by the gravitational pull of the Moon and the Sun.",
    "The Cretaceous–Paleogene extinction event ended the age of non-avian dinosaurs.",
    "The Geneva Conventions establish international standards for humanitarian treatment in war.",
]


B_EXT = [
    "class Stack:\n    def push(self, x):\n        self.items.append(x)",
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    "SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1;",
    "const arr = [1, 2, 3].map(x => x * 2);",
    "import torch\nx = torch.randn(10, 3)\nprint(x.shape)",
    "if user.is_admin:\n    grant_access()",
    "try:\n    int(x)\nexcept ValueError:\n    pass",
    "await asyncio.sleep(1)",
    "DROP TABLE IF EXISTS sessions;",
    "function isEven(n) { return n % 2 === 0; }",
    "pd.merge(df1, df2, on='id', how='inner')",
    "return sorted(items, key=lambda x: x.priority)",
    "BEGIN TRANSACTION;",
    "np.einsum('ij,jk->ik', A, B)",
    "for x in collection:\n    process(x)",
    'def __repr__(self) -> str:\n    return f"<User {self.id}>"',
    "cd /var/log && tail -f application.log",
    "from typing import Optional, List",
    "mat = np.zeros((3, 3), dtype=np.float32)",
    "git rebase -i HEAD~5",
    "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id;",
    "let isLoggedIn = !!localStorage.getItem('token');",
    "def is_palindrome(s):\n    return s == s[::-1]",
    "kubectl get pods --namespace=production",
    "os.makedirs(path, exist_ok=True)",
    "Object.keys(obj).filter(k => obj[k] != null)",
    'assert isinstance(value, (int, float)), f"Expected number, got {type(value).__name__}"',
    "tar -czf backup.tar.gz /etc/",
    "class APIResponse(TypedDict):\n    status: int\n    data: Optional[dict]",
    'printf("Hello, %s! You are %d years old.\\n", name, age);',
    'with conn.cursor() as cur:\n    cur.execute("SELECT 1")',
    "EXPLAIN ANALYZE SELECT * FROM events WHERE created_at > '2025-01-01';",
    "process.env.NODE_ENV = 'production';",
    "df.groupby('category').agg({'value': ['mean', 'std']})",
    "git stash pop",
    "let timer = setInterval(() => check(), 1000);",
    "model.fit(X_train, y_train, epochs=10, batch_size=32)",
    "await Promise.race([fetch(url), timeout(5000)])",
    "import functools\n@functools.lru_cache(maxsize=128)\ndef expensive(x):\n    return compute(x)",
    "chmod +x deploy.sh",
    "Array.from({length: 10}, (_, i) => i * i)",
    "MERGE INTO target USING source ON target.id = source.id WHEN MATCHED THEN UPDATE SET val = source.val;",
    "def tokenize(text: str) -> list[str]:\n    return text.split()",
    "for line in sys.stdin:\n    print(line.upper(), end='')",
    "console.log(JSON.stringify(data, null, 2));",
    'pkill -f "redis-server"',
    "data = json.loads(response.text)",
    "subprocess.run(['ls', '-la'], check=True, capture_output=True)",
    "CREATE INDEX idx_users_email ON users(email);",
    "with torch.no_grad():\n    output = model(input_tensor)",
    "git log --oneline --graph --all --decorate | head -20",
    "for i, val in enumerate(my_list):\n    print(i, val)",
    "const debounced = useMemo(() => debounce(handleSearch, 300), [handleSearch]);",
    "assert math.isclose(actual, expected, rel_tol=1e-9)",
    "aws s3 cp file.txt s3://my-bucket/",
    "def chunk(it, n):\n    return zip(*[iter(it)] * n)",
    "select count(distinct user_id) from sessions where ts >= now() - interval '1 day';",
    "df['log_value'] = np.log(df['value'].clip(lower=1e-9))",
    "let result = match input with | Some x -> x * 2 | None -> 0",
    "pip uninstall -y old_package",
    "with mock.patch('mymodule.requests.get') as m:\n    m.return_value.json.return_value = {}",
    'printf "%-20s %s\\n" "Name" "Email"',
    "from concurrent.futures import ThreadPoolExecutor",
    "Promise.allSettled(promises).then(results => results.forEach(log));",
    "redis-cli SET user:42:visits 1 EX 3600",
    "def levenshtein(a, b):\n    if not a: return len(b)\n    if not b: return len(a)",
    "app.use(express.json({ limit: '10mb' }));",
    "WITH recent AS (SELECT * FROM logs WHERE ts > NOW() - INTERVAL '1 hour') SELECT * FROM recent;",
    "pytest -v --cov=src tests/",
    "console.assert(arr.length > 0, 'array must not be empty');",
]


C_EXT = [
    "Türk halısı geleneksel olarak kadınlar tarafından dokunurdu.",
    "Akdeniz iklimi yazları sıcak ve kurak, kışları ılık ve yağışlı geçer.",
    "Edirne, uzun yıllar boyunca Osmanlı İmparatorluğu'nun başkenti olmuştur.",
    "Hakkımda söylediklerinize katılmadığımı belirtmek isterim.",
    "Bu yıl tatilimi Bodrum'da geçirmeyi düşünüyorum.",
    "Türkiye'nin en yüksek dağı Ağrı Dağı'dır.",
    "İstiklal Marşı 1921 yılında Mehmet Akif Ersoy tarafından yazılmıştır.",
    "Mantıyı genellikle yoğurt ve sarımsakla servis ederiz.",
    "Lütfen pencereyi kapatabilir misiniz, içerisi çok soğudu.",
    "Geçen sene İstanbul'a taşındığımdan beri kendimi çok daha iyi hissediyorum.",
    "Kara Kuvvetleri Komutanlığı Ankara'da bulunmaktadır.",
    "Üniversite sınavı Türkiye'deki en stresli sınavlardan biri olarak kabul edilir.",
    "Çay, Türk kültüründe sosyal hayatın vazgeçilmez bir parçasıdır.",
    "Bu konuyu daha sonra konuşalım, şimdi vaktim yok.",
    "Türk Hava Yolları, dünyanın en çok ülkeye uçan havayolu şirketlerinden biridir.",
    "Doğu Anadolu'da kışlar oldukça sert geçer.",
    "Pamukkale'nin beyaz traverten terasları UNESCO Dünya Mirası listesindedir.",
    "Babaannem her hafta bize ev yapımı börek getirirdi.",
    "Selçuklu mimarisi çini sanatı ile ünlüdür.",
    "Yarışmanın sonuçları yarın açıklanacakmış.",
    "Şehir merkezindeki trafik son yıllarda iyice yoğunlaştı.",
    "Hayatımda en çok pişman olduğum şey üniversiteye geç başlamış olmaktır.",
    "Türkiye dünyanın en büyük fındık üreticilerinden biridir.",
    "Ege kıyısındaki balık restoranları muhteşem manzaralara sahiptir.",
    "Yılbaşı gecesi ailemizle birlikte oturup eski filmler izledik.",
    "Kanuni Sultan Süleyman, Osmanlı tarihinin en uzun süre tahta kalan padişahlarından biridir.",
    "Bursa, Osmanlı Devleti'nin ilk başkenti olarak bilinir.",
    "Bu kelimeyi yanlış telaffuz ediyorsunuz, doğrusu farklı.",
    "Sınavlardan sonra arkadaşlarla buluşup birlikte film izledik.",
    "Türkçede ünlü uyumu önemli bir dilbilgisi kuralıdır.",
    "Kapadokya'da sıcak hava balonu turuna katılmak unutulmaz bir deneyimdi.",
    "Bu hafta sonu evde dinlenmeyi planlıyorum.",
    "Anadolu'da yapılan arkeolojik kazılar binlerce yıllık medeniyetleri ortaya çıkarmıştır.",
    "Maaşımı her ayın on beşinde alıyorum.",
    "Türk sineması son yıllarda uluslararası festivallerde başarılar elde etti.",
    "Akşam yemeği için ne hazırlayalım?",
    "Kayseri pastırması Türk mutfağının en bilinen lezzetlerindendir.",
    "Ekonomi dersini bu dönem ilk defa alıyorum.",
    "Karadeniz bölgesi yağmurlu iklimi ve yemyeşil doğasıyla tanınır.",
    "Yıllar geçtikçe arkadaşlarım birbirinden uzaklaştı.",
    "Eski Türk filmlerini hâlâ büyük zevkle izliyorum.",
    "Bilgisayarım birkaç gündür düzgün çalışmıyor.",
    "Cumhuriyet, 29 Ekim 1923'te ilan edilmiştir.",
    "Anneannemin Erzurum'dan getirdiği civil peyniri çok lezzetlidir.",
    "Bu sabah erken kalktım çünkü uçağımı kaçırmak istemiyordum.",
    "Mimar Sinan, Süleymaniye Camii'ni tasarlamıştır.",
    "Türkiye'de futbol en popüler spordur.",
    "Geçen hafta düzenlenen konferans büyük ilgi gördü.",
    "Bu adresi telefonunuza kaydeder misiniz?",
    "Konya ovası Türkiye'nin tahıl ambarı olarak bilinir.",
    "Çocukken anneannem bana Türk masalları okurdu.",
    "Restorandaki garson çok kibar ve ilgiliydi.",
    "Süleyman Demirel siyasi yaşamı boyunca dokuz kez başbakanlık yapmıştır.",
    "Kuzey Kıbrıs Türk Cumhuriyeti 1983 yılında kurulmuştur.",
    "Bu yağmurda dışarı çıkmak hiç akıllıca olmaz.",
    "Adana kebabı doğru pişirildiğinde nefis bir tat verir.",
    "Annenizin sağlığı nasıl, geçmiş olsun.",
    "Boğaziçi Köprüsü 1973 yılında hizmete açılmıştır.",
    "Ankara, Türkiye Cumhuriyeti'nin başkentidir.",
    "İlk maaşımla anneme küçük bir hediye almıştım.",
    "Karagöz ve Hacivat geleneksel Türk gölge tiyatrosunun başkahramanlarıdır.",
    "Üzgünüm ama bugün size yardım edemem.",
    "Ortaokul yıllarımda matematikten çok zorlanırdım.",
    "Çanakkale Boğazı'nda yapılan köprü Avrupa'yı Asya'ya bağlamaktadır.",
    "Hafta sonu ailem geliyor, şehri gezdireceğim.",
    "Türkçedeki \"evet\" ve \"hayır\" kelimeleri en sık kullanılanlar arasındadır.",
    "Babam emeklilikten sonra bahçe işleriyle uğraşmaya başladı.",
    "Aşure, Muharrem ayında geleneksel olarak yapılan tatlı bir yemektir.",
    "Dolmuşla şehirde dolaşmak hem ucuz hem pratik.",
    "Kar fırtınası nedeniyle okullar bir gün tatil edildi.",
]


D_EXT = [
    "The Licensee shall not sublicense, transfer, or assign any rights granted under this Agreement without the prior written consent of the Licensor.",
    "We employed Bayesian inference with weakly informative priors to estimate the posterior distribution.",
    "The Service Provider shall use commercially reasonable efforts to maintain at least 99.9% uptime.",
    "Treatment effects were heterogeneous across subgroups defined by baseline severity.",
    "The Borrower hereby pledges the Collateral as security for the punctual repayment of the Loan.",
    "Power analyses indicated that a sample size of 240 would yield 80% power at α = 0.05.",
    "Each party acknowledges that any breach of the confidentiality obligations will cause irreparable harm.",
    "The kinetic energy of a non-relativistic particle is one half the product of its mass and the square of its velocity.",
    "The Effective Date of this Agreement shall be the date of execution by the last party to sign.",
    "Robustness checks were performed by re-estimating the model with alternative measurement instruments.",
    "The Servicer shall remit collected payments to the Trustee on each Distribution Date.",
    "The functional form of the relationship was approximated using cubic B-splines with five interior knots.",
    "Subject to the terms hereof, the Issuer hereby grants to the Holder the right to convert the Note into shares.",
    "Western blotting confirmed the absence of the target protein in the knockout cell line.",
    "The Indemnifying Party shall promptly assume the defense of any Third-Party Claim with counsel reasonably satisfactory to the Indemnified Party.",
    "The Lyapunov function decreases monotonically along all trajectories of the closed-loop system.",
    "The Tenant shall surrender the Premises at the expiration of the Term in good order, reasonable wear and tear excepted.",
    "Reaction kinetics were modeled as second-order with respect to the limiting reagent.",
    "No assignment of this Agreement, whether by operation of law or otherwise, shall be valid without the express written consent of the non-assigning party.",
    "The proposed estimator achieves the Cramér–Rao bound asymptotically under mild regularity conditions.",
    "The Company shall maintain books and records in accordance with generally accepted accounting principles.",
    "Density functional theory calculations were performed using a 6-31G(d) basis set with the B3LYP functional.",
    "The remedies provided in this Section are cumulative and not exclusive of any other remedies available at law or equity.",
    "The microbial community composition was characterized using 16S rRNA gene amplicon sequencing.",
    "Force majeure shall not relieve any party of obligations that arose prior to the occurrence of the force majeure event.",
    "The variational lower bound on the marginal log-likelihood was optimized using stochastic gradient descent.",
    "The Pledgor hereby grants the Pledgee a continuing security interest in all Collateral now or hereafter acquired.",
    "Cells were fixed with 4% paraformaldehyde and permeabilized with 0.1% Triton X-100 prior to immunostaining.",
    "Each warranty made by the Seller is a separate, independent, and material representation, and survives Closing for two years.",
    "The asymptotic distribution of the test statistic was approximated by Monte Carlo simulation with ten thousand replications.",
    "Notice required or permitted hereunder shall be in writing and delivered by hand, certified mail, or overnight courier.",
    "The thermal conductivity of the composite material increased monotonically with filler volume fraction up to the percolation threshold.",
    "Title to the Goods shall pass to the Buyer upon delivery to the Buyer's designated freight carrier.",
    "We adopt a difference-in-differences design to identify the causal effect of the policy intervention.",
    "The Initial Term of this Agreement shall be three (3) years, automatically renewing for successive one-year terms unless terminated.",
    "Phylogenetic relationships were inferred using maximum likelihood with one thousand bootstrap replicates.",
    "The Distributor shall not, directly or indirectly, market or sell the Products outside the Territory without prior written approval.",
    "We pre-registered our hypotheses, sample size, and analysis plan on the Open Science Framework prior to data collection.",
    "The Pledged Securities shall be held in a segregated account in the name of the Pledgee.",
    "The activation energy of the reaction was determined from an Arrhenius plot of ln(k) versus 1/T.",
    "The parties acknowledge that they have read this Agreement, understand its terms, and have had the opportunity to consult counsel.",
    "The treatment group exhibited a statistically significant reduction in primary outcome measures relative to placebo.",
    "The Buyer's obligation to consummate the Transaction is subject to the satisfaction of each of the Closing Conditions set forth in Article V.",
    "Single-cell RNA sequencing revealed three transcriptionally distinct subpopulations within the tumor microenvironment.",
    "The arbitrator shall have the authority to award any remedy or relief that a court of competent jurisdiction could order.",
    "Convergence of the iterative algorithm was assessed using the relative change in parameter estimates between successive iterations.",
    "Each Party shall bear its own costs and expenses, including attorneys' fees, except as expressly provided herein.",
    "The Hamiltonian flow on the cotangent bundle preserves the symplectic form by Liouville's theorem.",
    "Notwithstanding any other provision of this Agreement, neither party shall be liable for indirect, special, or consequential damages.",
    "The data are consistent with a power-law decay with an exponent of approximately negative two.",
    "Upon the occurrence of an Event of Default, the Lender may declare the entire principal amount immediately due and payable.",
    "The error bars in Figure 3 represent the standard error of the mean across n = 12 biological replicates.",
    "The waiver of any right or remedy in any one instance shall not constitute a waiver of any other right or remedy.",
    "The CRISPR–Cas9 system was used to generate isogenic knockout lines of the gene of interest.",
    "The Receiving Party shall notify the Disclosing Party promptly upon discovery of any unauthorized disclosure of Confidential Information.",
    "The Reynolds number characterizes the relative importance of inertial and viscous forces in fluid flow.",
    "Each party warrants that it has all requisite legal capacity to enter into this Agreement and to perform its obligations hereunder.",
    "Co-immunoprecipitation experiments demonstrated a direct physical interaction between the two candidate binding partners.",
    "The Maximum Liability of either party for any claim arising under this Agreement shall not exceed the Fees paid in the preceding twelve months.",
    "The proposed mechanism is consistent with the observed isotope effect of approximately 6.7 at room temperature.",
    "The Effective Date shall be deemed to be the date on which the last of the Conditions Precedent has been satisfied or waived.",
    "Mass spectrometry analysis identified a previously uncharacterized post-translational modification at residue 137.",
    "The arbitration shall be conducted in the English language at a venue mutually agreed by the Parties.",
    "The receiver operating characteristic curve yielded an area under the curve of 0.87.",
    "No party shall make any public announcement concerning this Agreement without the prior written approval of the other party.",
    "The renormalization group flow exhibits a non-trivial fixed point in dimensions less than four.",
    "The expense reimbursement obligation shall survive any termination or expiration of this Agreement for a period of six (6) years.",
    "The phenotype was rescued by transgenic expression of the wild-type allele under the control of the endogenous promoter.",
    "The Carbon Footprint Statement attached as Exhibit C is incorporated herein by reference and forms an integral part of the Agreement.",
    "We document a strong negative correlation (r = −0.78, p < 0.001) between the two physiological markers.",
]


JBB_CACHE = Path("/workspace/nla-research/phase1_inputs_v2_tierE_jbb_sample.json")


def sample_tierE_from_jbb(seed: int = 42, per_category: int = 7) -> list[dict]:
    if JBB_CACHE.exists():
        cached = json.loads(JBB_CACHE.read_text())
        if cached.get("seed") == seed and cached.get("per_category") == per_category:
            return cached["items"]

    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]
    by_cat: dict[str, list[dict]] = {}
    for r in ds:
        by_cat.setdefault(r["Category"], []).append(dict(r))
    rng = random.Random(seed)
    sampled: list[dict] = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        sampled.extend(rng.sample(items, per_category))
    JBB_CACHE.write_text(json.dumps(
        {"seed": seed, "per_category": per_category,
         "source": "JailbreakBench/JBB-Behaviors:harmful",
         "items": sampled},
        ensure_ascii=False, indent=2,
    ))
    return sampled


def load_v1_inputs() -> list[dict]:
    rows = json.loads(Path("/workspace/nla-research/phase1_inputs.json").read_text())
    assert len(rows) == 150
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/workspace/nla-research/phase1_inputs_v2.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert len(A_EXT) == 70, f"A_EXT has {len(A_EXT)} (need 70)"
    assert len(B_EXT) == 70, f"B_EXT has {len(B_EXT)} (need 70)"
    assert len(C_EXT) == 70, f"C_EXT has {len(C_EXT)} (need 70)"
    assert len(D_EXT) == 70, f"D_EXT has {len(D_EXT)} (need 70)"

    e_jbb = sample_tierE_from_jbb(seed=args.seed, per_category=7)
    assert len(e_jbb) == 70, f"E_EXT has {len(e_jbb)} (need 70)"

    rows: list[dict] = list(load_v1_inputs())
    next_idx = 150
    for tier, ext, jbb in [
        ("A", A_EXT, None),
        ("B", B_EXT, None),
        ("C", C_EXT, None),
        ("D", D_EXT, None),
        ("E", None, e_jbb),
    ]:
        if ext is not None:
            for j, text in enumerate(ext):
                rows.append({
                    "idx": next_idx, "tier": tier, "tier_idx": 30 + j,
                    "text": text, "source": "v2_hardcoded",
                })
                next_idx += 1
        else:
            for j, item in enumerate(jbb):
                rows.append({
                    "idx": next_idx, "tier": tier, "tier_idx": 30 + j,
                    "text": item["Goal"], "source": "JBB-Behaviors:harmful",
                    "jbb_category": item["Category"],
                    "jbb_behavior": item["Behavior"],
                    "jbb_origin_source": item["Source"],
                })
                next_idx += 1

    assert len(rows) == 500, f"got {len(rows)}, need 500"
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} inputs to {args.out}")
    counts = {t: sum(1 for r in rows if r["tier"] == t) for t in "ABCDE"}
    print(f"per-tier counts: {counts}")
    print(f"JBB sample cached at: {JBB_CACHE}")


if __name__ == "__main__":
    main()
