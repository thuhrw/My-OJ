import os
import json
import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    status,
    Query,
    Depends,
    Cookie,
    Response,
    Request,
)
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from enum import Enum
import threading
from fastapi.exceptions import RequestValidationError
import subprocess
import tempfile
import psutil
import time
import signal
from typing import Tuple
from datetime import datetime
import bcrypt
import uuid
import sqlite3

DATABASE_PATH = "app.db"


def create_user_table():
    """创建用户表"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone() is not None
    if table_exists:
        conn.close()
        return
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            join_time TEXT NOT NULL,
            role TEXT NOT NULL,
            submit_count INTEGER DEFAULT 0,
            resolve_count INTEGER DEFAULT 0
        )
    """
    )
    conn.commit()
    conn.close()


app = FastAPI()
PROBLEMS_DIR = r"C:\Users\14395\Desktop\git\pa2-oj-2024010702\problems"
LANGUAGES_DIR = r"C:\Users\14395\Desktop\git\pa2-oj-2024010702\languages"
SUBMISSIONS_DIR = r"C:\Users\14395\Desktop\git\pa2-oj-2024010702\submissions"
TEMP_DIR = r"C:\Users\14395\Desktop\git\pa2-oj-2024010702\temp"
USERS_DIR = r"C:\Users\14395\Desktop\git\pa2-oj-2024010702\users"
SESSIONS_DIR = r"C:\Users\14395\Desktop\git\pa2-oj-2024010702\sessions"


def get_problem_path(problem_id):
    """获取题目文件路径"""
    return os.path.join(PROBLEMS_DIR, f"{problem_id}.json")


def get_language_path(language_name):
    """获取语言路径"""
    return os.path.join(LANGUAGES_DIR, f"{language_name}.json")


def get_submission_path(submission_id):
    """获取提交文件路径"""
    return os.path.join(SUBMISSIONS_DIR, f"{submission_id}.json")


@app.get("/")
async def welcome():
    return "Welcome!"


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    BANNED = "banned"


class User(BaseModel):
    user_id: int
    username: str
    join_time: str
    role: UserRole
    submit_count: int
    resolve_count: int


class UserCreate(BaseModel):  # 创建用户时所提交上来的数据结构
    username: str
    password: str


class LoginRequest(BaseModel):  # 登录时所提交上来的数据结构
    username: str
    password: str


def get_user_by_username(username: str):
    """根据用户名在数据库中寻找用户"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user_data = cursor.fetchone()
    conn.close()

    if user_data:
        return {
            "user_id": user_data[0],
            "username": user_data[1],
            "password_hash": user_data[2],
            "join_time": user_data[3],
            "role": user_data[4],
            "submit_count": user_data[5],
            "resolve_count": user_data[6],
        }
    return None


def get_user_by_user_id(user_id: str):
    """根据用户ID在数据库中寻找用户"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()

    if user_data:
        return {
            "user_id": user_data[0],
            "username": user_data[1],
            "password_hash": user_data[2],
            "join_time": user_data[3],
            "role": user_data[4],
            "submit_count": user_data[5],
            "resolve_count": user_data[6],
        }
    return None


def create_user(
    user_id: str, username: str, password: str, role: UserRole = UserRole.USER
):
    """在数据库中创建用户，利用要求的bcrypt的密码进行加密"""
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(
        "utf-8"
    )
    join_time = datetime.now().isoformat()

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (user_id, username, password_hash, join_time, role) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, join_time, role),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=400, detail="用户名已存在")
    finally:
        conn.close()

    return {
        "user_id": user_id,
        "username": username,
        "join_time": join_time,
        "role": role,
        "submit_count": 0,
        "resolve_count": 0,
    }


def init_admin_account():
    """建立初始管理员账号，如果已经存在就直接return，防止因为服务器重启而建立多个初始管理员"""
    admin_username = "admin"
    admin_password = "adminpassword"
    existing_admin = get_user_by_username(admin_username)
    if existing_admin:
        return

    user_id = get_next_user_id()

    if not get_user_by_username(admin_username):
        create_user(
            user_id=user_id,
            username=admin_username,
            password=admin_password,
            role=UserRole.ADMIN,
        )


def get_next_user_id():
    """本地实现自增用户id"""
    # TODO:因为一开始没用数据库，在设计submission提交的时候已经使用了这种方法，为了一致性选择本地实现自增用户id。这里可改进为使用数据库的自增id
    user_count_file = os.path.join(USERS_DIR, "user_count.txt")
    with open(user_count_file, "r+") as f:
        content = f.read().strip()
        next_id = int(content) + 1
        f.seek(0)
        f.truncate()
        f.write(str(next_id))
        return next_id


create_user_table()  # 创建用户表
init_admin_account()  # 初始化管理员


def get_current_user(request: Request):
    """通过当前状态，获取当前用户"""
    session_id = request.cookies.get("session_id")

    if not session_id:
        raise HTTPException(status_code=401, detail="用户未登录")

    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_file):
        raise HTTPException(status_code=401, detail="Invalid session")

    with open(session_file, "r") as f:
        session_data = json.load(f)

    # 验证用户是否存在
    user_data = get_user_by_username(session_data["username"])
    if not user_data:
        raise HTTPException(status_code=404, detail="用户不存在")

    return User(**user_data)


def is_admin(user: User = Depends(get_current_user)):
    """检查是否是管理员"""
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="用户无权限")
    return user


@app.post("/api/auth/login")
async def login(response: Response, request: dict):
    """用户登录"""
    try:
        request = LoginRequest(**request)
    except ValidationError:
        raise HTTPException(status_code=400, detail="参数错误")

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (request.username,))
    user_data = cursor.fetchone()
    conn.close()

    # 验证用户是否存在
    if not user_data:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    user_id, username, password_hash, join_time, role, submit_count, resolve_count = (
        user_data
    )

    if not bcrypt.checkpw(
        request.password.encode("utf-8"), password_hash.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if role == UserRole.BANNED:
        raise HTTPException(status_code=403, detail="用户被禁用")

    session_id = str(uuid.uuid4())
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    with open(session_file, "w") as f:
        json.dump(
            {
                "user_id": user_id,
                "username": username,
                "role": role,
                "created_at": datetime.now().isoformat(),
            },
            f,
        )

    response.set_cookie(key="session_id", value=session_id, httponly=True)

    return {
        "code": 200,
        "msg": "login success",
        "data": {
            "user_id": user_id,
            "username": username,
            "role": role,
        },
    }


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    """用户登出"""
    # 直接从cookie获取session_id
    session_id = request.cookies.get("session_id")

    if session_id:
        session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
        else:
            raise HTTPException(status_code=401, detail="未登录")
    else:
        raise HTTPException(status_code=401, detail="未登录")

    # 清除客户端cookie
    response.delete_cookie(key="session_id")

    return {"code": 200, "msg": "logout success", "data": "null"}


@app.post("/api/users/admin")
async def create_admin_user(request: dict, current_user: User = Depends(is_admin)):
    """管理员才可以添加新的管理员"""
    try:
        request = UserCreate(**request)
    except ValidationError:
        raise HTTPException(status_code=400, detail="参数错误")

    user_id = get_next_user_id()

    password_hash = bcrypt.hashpw(
        request.password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    join_time = datetime.now().isoformat()

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO users 
               (user_id, username, password_hash, join_time, role) 
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, request.username, password_hash, join_time, UserRole.ADMIN),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=400, detail="用户名已存在")
    finally:
        conn.close()

    return {
        "code": 200,
        "msg": "success",
        "data": {"user_id": user_id, "username": request.username},
    }


@app.post("/api/users/")
async def create_user(request: dict):
    """创建新用户"""
    try:
        request = UserCreate(**request)
    except ValidationError:
        raise HTTPException(status_code=400, detail="参数错误")

    user_id = get_next_user_id()

    password_hash = bcrypt.hashpw(
        request.password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")
    join_time = datetime.now().isoformat()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """INSERT INTO users 
               (user_id, username, password_hash, join_time, role) 
               VALUES (?, ?, ?, ?, ?)""",
            (
                user_id,
                request.username,
                password_hash,
                join_time,
                UserRole.USER,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail="用户名已存在")
    finally:
        conn.close()

    return {
        "code": 200,
        "msg": "success",
        "data": {"user_id": user_id, "username": request.username},
    }


@app.get("/api/users/{user_id}")
async def get_user(user_id: str, current_user: User = Depends(get_current_user)):
    """查询某一用户信息"""
    if not (current_user.user_id == user_id or current_user.role == UserRole.ADMIN):
        raise HTTPException(status_code=403, detail="用户无权限")

    user_data = get_user_by_user_id(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="用户不存在")

    return {
        "code": 200,
        "msg": "success",
        "data": User(**user_data).model_dump(),
    }


@app.put("/api/users/{user_id}/role")
async def update_user_role(
    user_id: str, role: UserRole, current_user: User = Depends(get_current_user)
):
    """更新用户角色"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="用户无权限")

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()
    if not user_data:
        conn.close()
        raise HTTPException(status_code=404, detail="用户不存在")

    cursor.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user_id))
    conn.commit()
    conn.close()

    return {
        "code": 200,
        "msg": "role updated",
        "data": {"user_id": user_id, "role": role},
    }


@app.get("/api/users/")
async def list_users(page: int, page_size: int, current_user: User = Depends(is_admin)):
    """获取用户列表"""
    if page < 1:
        raise HTTPException(status_code=400, detail="参数错误")

    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="参数错误")

    offset = (page - 1) * page_size

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:

        cursor.execute("SELECT COUNT(*) FROM users")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT user_id, username, join_time, role, submit_count, resolve_count "
            "FROM users "
            "ORDER BY join_time DESC "  # 按注册时间倒序排列
            "LIMIT ? OFFSET ?",
            (page_size, offset),
        )
        users_data = cursor.fetchall()

    finally:
        conn.close()

    users = []
    for user in users_data:
        users.append(
            {
                "user_id": user[0],
                "username": user[1],
                "join_time": user[2],
                "submit_count": user[4],
                "resolve_count": user[5],
            }
        )

    return {"code": 200, "msg": "success", "data": {"total": total, "users": users}}


# 数据模型
class Sample(BaseModel):  # 样例类
    input: str
    output: str


class TestStatus(str, Enum):  # 测试点状态类
    ACCEPTED = "Accepted"
    WRONG_ANSWER = "Wrong Answer"
    TIME_LIMIT_EXCEEDED = "Time Limit Exceeded"
    MEMORY_LIMIT_EXCEEDED = "Memory Limit Exceeded"
    RUNTIME_ERROR = "Runtime Error"
    COMPILATION_ERROR = "Compilation Error"
    UNK = "Unknown Error"


class Result(BaseModel):  # 单个测试点返回的评测结果类
    caseid: int
    status: TestStatus
    time: Optional[float] = None  # 单个测试用例的运行时间
    memory: Optional[float] = None  # 单个测试用例的内存


class TestCase(BaseModel):  # 测试用例类
    input: str
    output: str


class Problem(BaseModel):  # 题目类
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str
    input_description: str
    output_description: str
    samples: List[Sample]
    constraints: str
    testcases: List[TestCase]
    time_limit: float = 1.0  # 默认时间限制为1秒
    memory_limit: float = 128.0  # 默认内存限制为128MB
    hint: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    difficulty: Optional[str] = None


@app.get("/api/problems/")
async def list_problems():
    """获取题目列表"""
    problems = []
    for filename in os.listdir(PROBLEMS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(PROBLEMS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                problems.append({"id": data["id"], "title": data["title"]})
    return {"code": 200, "msg": "success", "data": problems}


@app.post("/api/problems/")
async def create_problem(problem: dict, user: User = Depends(get_current_user)):
    """创建新题目"""
    try:
        problem = Problem(**problem)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="字段缺失或格式错误")
    file_path = get_problem_path(problem.id)
    if os.path.exists(file_path):
        raise HTTPException(status_code=409, detail="id已存在")

    problem_data = problem.model_dump()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(problem_data, f)

    return {"code": 200, "msg": "add success", "data": {"id": problem.id}}


@app.delete("/api/problems/{problem_id}")
async def delete_problem(problem_id: str, user: User = Depends(is_admin)):
    """删除题目"""
    file_path = get_problem_path(problem_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="题目不存在")

    os.remove(file_path)
    return {"code": 200, "msg": "delete success", "data": {"id": problem_id}}


@app.get("/api/problems/{problem_id}")
async def get_problem(problem_id: str):
    """获取题目详情"""
    file_path = get_problem_path(problem_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="题目不存在")

    with open(file_path, "r", encoding="utf-8") as f:
        return {"code": 200, "msg": "success", "data": json.load(f)}


class Language(BaseModel):  # 语言类
    name: str
    file_ext: str  # 代码文件扩展名
    compile_cmd: Optional[str] = None  # 编译命令
    run_cmd: str  # 运行命令
    source_template: Optional[str] = None  # 代码模板
    time_limit: Optional[float] = None
    memory_limit: Optional[int] = None


@app.post("/api/languages/", status_code=status.HTTP_201_CREATED)
async def register_language(request: dict, current_user: User = Depends(is_admin)):
    """注册新编程语言"""

    try:
        request = Language(**request)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="参数错误")

    file_path = get_language_path(request.name)
    lang_config = request.model_dump()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(lang_config, f)

    return {
        "code": 200,
        "msg": "language registered",
        "data": {"name": request.name},
    }


@app.get("/api/languages/")
async def list_languages():
    """获取已注册的编程语言列表"""
    languages = {"name": []}
    for filename in os.listdir(LANGUAGES_DIR):
        if filename.endswith(".json"):
            path = os.path.join(LANGUAGES_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                languages["name"].append(data["name"])
    return {"code": 200, "msg": "success", "data": languages}


class SubmissionStatus(str, Enum):  # 提交的状态类
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


class CreateSubmission(BaseModel):  # 创建提交的数据结构
    problem_id: str
    language: str
    code: str


class Submission(BaseModel):  # 提交类
    id: int
    problem_id: str
    user_id: int
    language: str
    code: str
    status: SubmissionStatus
    teststatus: List[Result] = []  # 测试结果列表
    score: int = 10
    count: int


class JudgeResult(BaseModel):  # 评测结果类
    teststatus: List[Result] = []
    status: SubmissionStatus
    count: int


def run_process(
    command: List[str],
    input_data: str,
    time_limit: float,
    memory_limit: int,
    work_dir: str,
) -> tuple[str, str, float, float, bool]:
    """运行进程并监控资源使用"""
    max_memory = 0  # 内存峰值（字节）
    timed_out = False
    memory_exceeded = False
    start_time = time.time()

    def memory_monitor(process_pid):
        nonlocal max_memory, memory_exceeded
        try:
            proc = psutil.Process(process_pid)
            while True:
                try:
                    mem_info = proc.memory_info()
                    current_mem = mem_info.rss  # 常驻内存大小

                    if current_mem > max_memory:
                        max_memory = current_mem

                    # 检查内存是否超限
                    if current_mem > memory_limit * 1024 * 1024:
                        memory_exceeded = True
                        os.kill(process_pid, signal.SIGTERM)
                        return

                    # 检查进程是否结束
                    if not proc.is_running():
                        return

                    time.sleep(0.01)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return
        except Exception:
            pass

    try:
        # 启动进程
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        # 启动内存监控线程
        monitor_thread = threading.Thread(target=memory_monitor, args=(process.pid,))
        monitor_thread.daemon = True
        monitor_thread.start()

        # 向进程提供输入并获取输出
        try:
            stdout, stderr = process.communicate(
                input=input_data, timeout=time_limit + 0.5
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            stdout, stderr = process.communicate()

        # 等待监控线程结束
        monitor_thread.join(timeout=0.1)

        # 计算运行时间
        run_time = time.time() - start_time

        # 计算内存使用（MB）
        memory_used = max_memory / (1024 * 1024) if max_memory > 0 else 0

        return stdout, stderr, run_time, memory_used, timed_out or memory_exceeded

    except Exception as e:
        raise RuntimeError(f"Process execution failed: {str(e)}")


def normalize_output(output: str):
    """标准化输出：移除末尾空白和多余换行"""
    # 移除每行末尾空白
    lines = [line.rstrip() for line in output.splitlines()]
    # 移除末尾空行
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def compare_output(actual: str, expected: str):
    """比较两个输出是否一致"""
    return normalize_output(actual) == normalize_output(expected)


def judge_submission(problem: Problem, code: str, language: str):
    """评测提交的代码"""
    lang_path = get_language_path(language)
    if not os.path.exists(lang_path):
        test_status = []
        for i in range(len(problem.testcases)):
            test_status.append(Result(i + 1, TestStatus.UNK))
        return JudgeResult(
            teststatus=test_status, count=0, status=SubmissionStatus.SUCCESS
        )

    with open(lang_path, "r", encoding="utf-8") as f:
        lang_config = Language(**json.load(f))

    # 创建临时工作目录
    with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
        work_dir = temp_dir  # 直接使用字符串路径

        # 保存源代码
        source_path = os.path.join(work_dir, f"main.{lang_config.file_ext}")
        with open(source_path, "w") as f:
            f.write(code)

        executable_path = None
        if lang_config.compile_cmd:
            # 编译代码
            executable_path = os.path.join(work_dir, "main")
            # 格式化编译命令
            compile_cmd = lang_config.compile_cmd.format(
                src=source_path, exe=executable_path
            ).split()

            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(work_dir),
                    text=True,
                    timeout=10,  # 编译超时10秒
                )
                if compile_result.returncode != 0:
                    test_status = []
                    for i in range(len(problem.testcases)):
                        test_status.append(Result(i + 1, TestStatus.COMPILATION_ERROR))

                    return JudgeResult(
                        teststatus=test_status,
                        status=SubmissionStatus.SUCCESS,
                        count=0,
                    )
            except subprocess.TimeoutExpired:
                test_status = []
                for i in range(len(problem.testcases)):
                    test_status.append(Result(i + 1, TestStatus.COMPILATION_ERROR))
                return JudgeResult(
                    teststatus=test_status,
                    status=SubmissionStatus.SUCCESS,
                    count=0,
                )
            except Exception as e:
                test_status = []
                for i in range(len(problem.testcases)):
                    test_status.append(Result(i + 1, TestStatus.COMPILATION_ERROR))
                return JudgeResult(
                    teststatus=test_status,
                    status=SubmissionStatus.SUCCESS,
                    count=0,
                )

        # 准备运行命令
        run_cmd = lang_config.run_cmd.format(
            src=source_path, exe=executable_path if executable_path else ""
        ).split()

        # 使用题目限制或语言默认限制
        time_limit = min(
            problem.time_limit, lang_config.time_limit
        )  # 选择题目限时和语言限时中最小的
        memory_limit = min(
            problem.memory_limit, lang_config.memory_limit
        )  # 选择题目内存限制和语言内存限制中最小的

        total_score = 0
        test_status = []
        for idx, test_case in enumerate(problem.testcases):

            result = Result(caseid=idx + 1, status=TestStatus.ACCEPTED)

            try:
                # 运行并监控进程
                stdout, stderr, run_time, memory_used, resource_exceeded = run_process(
                    run_cmd, test_case.input, time_limit, memory_limit, work_dir
                )

                # 检查资源使用情况
                if resource_exceeded:
                    if memory_used > memory_limit:
                        result.status = TestStatus.MEMORY_LIMIT_EXCEEDED
                        result.memory = memory_used
                        result.time = run_time

                    else:
                        result.status = TestStatus.TIME_LIMIT_EXCEEDED
                        result.time = run_time
                        result.memory = memory_used

                    test_status.append(result)

                    continue

                # 检查运行结果
                if not compare_output(stdout, test_case.output):
                    result.status = TestStatus.WRONG_ANSWER
                    result.time = run_time
                    result.memory = memory_used
                    test_status.append(result)
                    continue

                # 测试用例通过
                total_score += 10  # 每个测试用例10分
                result.time = run_time
                result.memory = memory_used

            except Exception as e:
                result.status = TestStatus.RUNTIME_ERROR

            test_status.append(result)

        return JudgeResult(
            teststatus=test_status,
            status=SubmissionStatus.SUCCESS,
            count=total_score,
        )


def run_judge_in_background(submission_problem_id: str, submission_id: str):
    """后台运行评测任务"""
    problem_path = get_problem_path(submission_problem_id)
    submission_path = get_submission_path(submission_id)
    print(submission_path)
    with open(problem_path, "r+", encoding="utf-8") as f:
        problem = Problem(**json.load(f))

    with open(submission_path, "r+", encoding="utf-8") as f:
        submission = Submission(**json.load(f))

    result = judge_submission(problem, submission.code, submission.language)
    submission.status = result.status
    submission.count = result.count
    submission.teststatus = result.teststatus

    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(submission.model_dump(), f)


@app.post("/api/submissions/")
async def submit_code(request: dict, current_user: User = Depends(get_current_user)):
    """提交代码"""

    try:
        request = CreateSubmission(**request)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())

    file_path = get_problem_path(request.problem_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="题目不存在")

    file = open(
        r"C:\Users\14395\Desktop\git\pa2-oj-2024010702\app\count.txt",
        "r+",
        encoding="utf-8",
    )
    content = int(file.read())
    file.seek(0)
    file.truncate()
    content += 1

    # 创建提交记录
    submission = Submission(
        id=content,
        problem_id=request.problem_id,
        user_id=current_user.user_id,
        language=request.language,
        code=request.code,
        status=SubmissionStatus.PENDING,
        score=10,
        count=0,
    )
    file.write(str(content))
    file.close()
    file_path = get_submission_path(content)
    sub_config = submission.model_dump()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sub_config, f)

    # 启动后台评测任务
    threading.Thread(
        target=run_judge_in_background, args=(submission.problem_id, submission.id)
    ).start()

    return {
        "code": 200,
        "msg": "success",
        "data": {"submission_id": content, "status": submission.status},
    }


@app.get("/api/submissions/{submission_id}")
async def get_submission_result(
    submission_id: str, current_user: User = Depends(get_current_user)
):
    """获取评测结果"""
    submission_path = get_submission_path(submission_id)
    if not os.path.exists(submission_path):
        raise HTTPException(status_code=404, detail="评测不存在")

    with open(submission_path, "r", encoding="utf-8") as f:
        submission = Submission(**json.load(f))

    if (
        submission.user_id != current_user.user_id
        and not current_user.role == UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="权限不足")

    return {
        "code": 200,
        "msg": "success",
        "data": {
            "status": submission.status,
            "score": submission.score,
            "count": submission.count,
        },
    }


def get_submissions(
    user_id: Optional[int] = None,
    problem_id: Optional[str] = None,
    status: Optional[SubmissionStatus] = None,
    page: int = 1,
    page_size: int = 20,
) -> Tuple[int, List[Submission]]:
    """获取提交列表"""
    submissions = []
    for filename in os.listdir(SUBMISSIONS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(SUBMISSIONS_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    submissions.append(Submission(**data))
            except:
                continue

    filtered = []
    for sub in submissions:
        if user_id is not None and sub.user_id != user_id:
            continue
        if problem_id is not None and sub.problem_id != problem_id:
            continue
        if status is not None and sub.status != status:
            continue
        filtered.append(sub)

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    return total, filtered[start:end]


@app.get("/api/submissions/")
async def list_submissions(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    problem_id: Optional[str] = Query(None, description="Filter by problem ID"),
    status: Optional[SubmissionStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
):
    """获取提交列表"""

    if current_user.user_id != user_id:
        user_id = current_user.id

    total, submissions = get_submissions(
        user_id=user_id,
        problem_id=problem_id,
        status=status,
        page=page,
        page_size=page_size,
    )

    output = []
    for sub in submissions:
        if sub.status != SubmissionStatus.SUCCESS.value:
            output.append({"id": sub.id, "status": sub.status})
        else:
            output.append(
                {
                    "submission_id": sub.id,
                    "status": sub.status,
                    "score": sub.score,
                    "count": sub.count,
                }
            )
    return {
        "code": 200,
        "msg": "success",
        "data": {
            "total": total,
            "submissions": output,
        },
    }


@app.put("/api/submissions/{submission_id}/rejudge")
async def rejudge_submission(
    submission_id: str, current_user: User = Depends(is_admin)
):
    """重新评测"""

    file_path = get_submission_path(submission_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Submission not found")

    with open(file_path, "r+", encoding="utf-8") as f:
        submission = Submission(**json.load(f))

    newsubmission = Submission(
        id=submission.id,
        problem_id=submission.problem_id,
        user_id=submission.user_id,
        language=submission.language,
        code=submission.code,
        status=SubmissionStatus.PENDING,
        score=10,
        count=0,
    )
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(newsubmission.model_dump(), f)

    threading.Thread(
        target=run_judge_in_background, args=(submission.problem_id, submission.id)
    ).start()
    return {
        "code": 200,
        "msg": "rejudge started",
        "data": {"submission_id": newsubmission.id, "status": newsubmission.status},
    }
