import json

def generate_jsonl(file_path: str, num_records: int):
    with open(file_path, 'w',encoding='utf-8') as f:
        for i in range(num_records):
            record = {
                "title": f"Title {i}",
                "pub_time": "2025-04-27",
                "source": "niuniu瞎编",
                "content": f"4月25日召开的中共中央政治局会议强调，培育壮大新质生产力，打造一批新兴支柱产业。“高质量发展离不开创新驱动和产业支撑”“科技创新和产业创新，是发展新质生产力的基本路径”。今年一季度，我国规模以上高技术制造业增加值同比增长9.7%，信息传输、软件和信息技术服务业增加值增长10.3%"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
generate_jsonl("../test_data.jsonl", 100000)
