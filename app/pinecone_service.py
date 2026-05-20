from pinecone import Pinecone, ServerlessSpec
from app.utils import text_to_embedding
import time


def init_pinecone(api_key: str, index_name: str, dimension: int = 768):
    """Initialize Pinecone and ensure index exists."""
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)


def upsert_historical_cases(index):
    """
    Seed Pinecone with synthetic historical WIM (Weigh-in-Motion) cases.
    Each case is a real-world-like scenario from Pakistani highways.
    Call this once to populate the DB.
    """
    cases = [
        {
            "id": "case_001",
            "text": "Heavy truck class. 6 axles. Cargo extension detected: True. Signals: severe suspension sag, rear tyre bulge, cargo exceeding chassis height by 40%. Risk score: 92. Overloaded sugarcane truck on M-2 motorway.",
            "metadata": {
                "truck_class": "heavy",
                "axle_count": 6,
                "cargo_extension": True,
                "overload_label": "CRITICAL",
                "risk_score": 92,
                "location": "M-2 Motorway",
                "cargo_type": "sugarcane",
                "outcome": "Vehicle stopped, fined Rs. 50,000, cargo partially offloaded",
            },
        },
        {
            "id": "case_002",
            "text": "Extra-heavy truck. 8 axles. Cargo extension: True. Signals: extreme height, tyre deformation, chassis bending visible. Risk score: 97. Overloaded cement mixer near Lahore.",
            "metadata": {
                "truck_class": "extra-heavy",
                "axle_count": 8,
                "cargo_extension": True,
                "overload_label": "CRITICAL",
                "risk_score": 97,
                "location": "Lahore Ring Road",
                "cargo_type": "cement",
                "outcome": "Vehicle impounded, driver arrested",
            },
        },
        {
            "id": "case_003",
            "text": "Medium truck. 4 axles. Cargo extension: False. Signals: slightly uneven load distribution. Risk score: 35. Cotton bales within limits on N-55.",
            "metadata": {
                "truck_class": "medium",
                "axle_count": 4,
                "cargo_extension": False,
                "overload_label": "MEDIUM",
                "risk_score": 35,
                "location": "N-55 National Highway",
                "cargo_type": "cotton bales",
                "outcome": "Allowed passage after inspection",
            },
        },
        {
            "id": "case_004",
            "text": "Light truck. 2 axles. Cargo extension: False. Signals: none visible, load within bounds. Risk score: 12. Small goods carrier Karachi port.",
            "metadata": {
                "truck_class": "light",
                "axle_count": 2,
                "cargo_extension": False,
                "overload_label": "LOW",
                "risk_score": 12,
                "location": "Karachi Port Access Road",
                "cargo_type": "packaged goods",
                "outcome": "Cleared immediately",
            },
        },
        {
            "id": "case_005",
            "text": "Heavy truck. 6 axles. Cargo extension: True. Signals: cargo height 3m above chassis, side spillage, tyre contact abnormal. Risk score: 78. Overloaded bricks truck on GT Road.",
            "metadata": {
                "truck_class": "heavy",
                "axle_count": 6,
                "cargo_extension": True,
                "overload_label": "HIGH",
                "risk_score": 78,
                "location": "GT Road Rawalpindi",
                "cargo_type": "bricks",
                "outcome": "Stopped, weighed at static station, fined",
            },
        },
        {
            "id": "case_006",
            "text": "Medium truck. 4 axles. Cargo extension: False. Signals: mild suspension compression, load near limits. Risk score: 55. Fertilizer bags on M-3 motorway.",
            "metadata": {
                "truck_class": "medium",
                "axle_count": 4,
                "cargo_extension": False,
                "overload_label": "MEDIUM",
                "risk_score": 55,
                "location": "M-3 Motorway",
                "cargo_type": "fertilizer",
                "outcome": "Directed to weigh station for measurement",
            },
        },
        {
            "id": "case_007",
            "text": "Extra-heavy truck. 10 axles. Cargo extension: True. Signals: extremely overloaded, frame stress visible, tyres severely deformed. Risk score: 99. Steel coils transport near Faisalabad.",
            "metadata": {
                "truck_class": "extra-heavy",
                "axle_count": 10,
                "cargo_extension": True,
                "overload_label": "CRITICAL",
                "risk_score": 99,
                "location": "Faisalabad Industrial Zone",
                "cargo_type": "steel coils",
                "outcome": "Emergency stop, road closed temporarily",
            },
        },
        {
            "id": "case_008",
            "text": "Light truck. 2 axles. Cargo extension: False. Signals: load well distributed, no visible stress. Risk score: 8. Vegetables transport near Multan.",
            "metadata": {
                "truck_class": "light",
                "axle_count": 2,
                "cargo_extension": False,
                "overload_label": "LOW",
                "risk_score": 8,
                "location": "Multan Bypass",
                "cargo_type": "vegetables",
                "outcome": "Cleared immediately",
            },
        },
        {
            "id": "case_009",
            "text": "Heavy truck. 6 axles. Cargo extension: True. Signals: cargo piled 2m above roof, unsecured load risk. Risk score: 83. Garbage / debris haul in Islamabad.",
            "metadata": {
                "truck_class": "heavy",
                "axle_count": 6,
                "cargo_extension": True,
                "overload_label": "CRITICAL",
                "risk_score": 83,
                "location": "Islamabad Expressway",
                "cargo_type": "construction debris",
                "outcome": "Stopped and load secured before passage",
            },
        },
        {
            "id": "case_010",
            "text": "Medium truck. 4 axles. Cargo extension: False. Signals: tyre pressure slightly low, load acceptable. Risk score: 28. Animal feed bags on N-25.",
            "metadata": {
                "truck_class": "medium",
                "axle_count": 4,
                "cargo_extension": False,
                "overload_label": "LOW",
                "risk_score": 28,
                "location": "N-25 Balochistan",
                "cargo_type": "animal feed",
                "outcome": "Cleared with advisory",
            },
        },
    ]

    vectors = []
    for case in cases:
        embedding = text_to_embedding(case["text"])
        vectors.append({
            "id": case["id"],
            "values": embedding,
            "metadata": {**case["metadata"], "description": case["text"]},
        })

    index.upsert(vectors=vectors)
    return len(vectors)


def query_similar_cases(index, query_text: str, top_k: int = 3) -> list[dict]:
    """Embed query text and retrieve top-K similar historical cases."""
    query_embedding = text_to_embedding(query_text)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    return results.get("matches", [])


def check_index_populated(index) -> bool:
    """Return True if index already has vectors."""
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0) > 0
