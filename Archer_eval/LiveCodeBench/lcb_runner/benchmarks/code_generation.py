import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
import time
from datasets import load_dataset
from huggingface_hub import hf_hub_download


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


# ============== Fallback: hf_hub_download ==============

RELEASE_FILES = {
    "release_v1": ["test1.jsonl"],
    "release_v2": ["test1.jsonl", "test2.jsonl"],
    "release_v3": ["test1.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test1.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test1.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test6.jsonl"],  # v6 只需要 test6.jsonl (2025年后数据)
}


def _load_via_hf_hub_download(repo_id: str, release_version: str) -> list[dict]:
    """Fallback: 使用 hf_hub_download 下载 jsonl 文件"""
    files = RELEASE_FILES.get(release_version, ["test1.jsonl"])
    print(f"Fallback: Downloading {files} from {repo_id} via hf_hub_download...")
    
    # 尝试不同的 revision（分支名）
    revisions_to_try = [release_version, "main", "release_latest"]
    
    all_data = []
    for filename in files:
        downloaded = False
        for revision in revisions_to_try:
            try:
                print(f"  Trying {filename} with revision={revision}...")
                path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, revision=revision)
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
                downloaded = True
                print(f"  Successfully downloaded {filename} from revision={revision}")
                break
            except Exception as e:
                print(f"  Failed {filename} with revision={revision}: {e}")
                continue
        if not downloaded:
            print(f"  Warning: Could not download {filename} from any revision")
    return all_data


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    repo_id = "livecodebench/code_generation_lite"
    dataset = None
    
    # 尝试多种配置名（缓存里的配置名可能与请求的不同）
    configs_to_try = [
        {"name": "release_latest", "version_tag": release_version},
        {"version_tag": release_version},
        {"name": release_version},
        {},
    ]
    
    for config in configs_to_try:
        try:
            print(f"Trying load_dataset with config: {config}")
            dataset = load_dataset(repo_id, split="test", trust_remote_code=True, **config)
            dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
            print(f"Successfully loaded with config: {config}")
            break
        except Exception as e:
            print(f"Failed with config {config}: {e}")
            continue
    
    # 如果所有配置都失败，尝试 hf_hub_download
    if dataset is None:
        print(f"All load_dataset attempts failed, trying hf_hub_download fallback...")
        raw_data = _load_via_hf_hub_download(repo_id, release_version)
        start_time = time.time()
        dataset = [CodeGenerationProblem(**p) for p in raw_data]
        end_time = time.time()
        print(f"hf_hub_download time taken: {end_time - start_time} seconds")
    
    print(f"Filtering problems...")
    start_time = time.time()
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    end_time = time.time()
    print(f"Loaded {len(dataset)} problems in {end_time - start_time} seconds")
    
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    repo_id = "livecodebench/code_generation"
    
    try:
        dataset = load_dataset(repo_id, split="test")
        dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    except Exception as e:
        print(f"load_dataset failed: {e}")
        raw_data = _load_via_hf_hub_download(repo_id, release_version)
        dataset = [CodeGenerationProblem(**p) for p in raw_data]
    
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
