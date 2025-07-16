import streamlit as st
import requests
import json
import time
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
if "problem_id" not in st.session_state:
    st.session_state.problem_id = ""
if "submission_id" not in st.session_state:
    st.session_state.submission_id = ""


def api_request(method, endpoint, data=None, files=None, headers=None):
    """请求后端api"""
    url = f"{API_BASE_URL}{endpoint}"
    cookies = {"session_id": st.session_state.get("session_id", "")}

    try:
        if method == "GET":
            response = requests.get(url, params=data, cookies=cookies, headers=headers)
        elif method == "POST":
            if files:
                response = requests.post(
                    url, data=data, files=files, cookies=cookies, headers=headers
                )
            else:
                response = requests.post(
                    url, json=data, cookies=cookies, headers=headers
                )
        elif method == "PUT":
            response = requests.put(url, json=data, cookies=cookies, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, json=data, cookies=cookies, headers=headers)

        response.raise_for_status()
        return {"data": response.json(), "headers": response.headers}
    except requests.exceptions.HTTPError as e:
        return {
            "data": {
                "code": response.json().get("code"),
                "message": response.json().get("msg"),
            },
            "headers": response.headers,
        }


def navigate(page, problem_id=None, submission_id=None):
    "跳转到不同的页面"
    st.session_state.current_page = page
    if problem_id:
        st.session_state.problem_id = problem_id
    if submission_id:
        st.session_state.submission_id = submission_id
    st.rerun()


def login_page():
    "用户登录页面"
    st.title("用户登录")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("登录")
        username = st.text_input("用户名", key="login_username")
        password = st.text_input("密码", type="password", key="login_password")

        if st.button("登录"):
            if not username or not password:
                st.error("请输入用户名和密码")
            else:
                response = api_request(
                    "POST",
                    "/api/auth/login",
                    {"username": username, "password": password},
                )
                if response["data"]["code"] != 200:
                    st.error(response["data"]["message"])
                else:
                    data = response["data"]["data"]

                    st.session_state.logged_in = True
                    st.session_state.username = data["username"]
                    st.session_state.user_id = data["user_id"]
                    st.session_state.role = data["role"]
                    st.session_state.session_id = (
                        response["headers"]
                        .get("Set-Cookie", "")
                        .split(";")[0]
                        .split("=")[1]
                    )
                    navigate("problems")
                    st.success("登录成功")

    with col2:
        st.subheader("注册")
        reg_username = st.text_input("用户名", key="reg_username")
        reg_password = st.text_input("密码", type="password", key="reg_password")

        if st.button("注册"):
            if not reg_username or not reg_password:
                st.error("请输入用户名和密码")
            else:
                response = api_request(
                    "POST",
                    "/api/users/",
                    {"username": reg_username, "password": reg_password},
                )
                if response["data"]["code"] == 200:
                    st.success("注册成功，请登录")
                else:
                    st.error(response["data"]["message"])


def problems_page():
    "题目列表页面"
    st.title("题目列表")
    response = api_request("GET", "/api/problems/")
    if response["data"]["code"] == 200:
        problems = response["data"]["data"]
        if not problems:
            st.info("暂无题目")
        else:
            cols = st.columns(3)
            cols[0].write("**ID**")
            cols[1].write("**标题**")
            cols[2].write("**操作**")
            for problem in problems:
                cols = st.columns(3)
                cols[0].write(problem["id"])
                cols[1].write(problem["title"])
                if cols[2].button("查看", key=f"view_{problem['id']}"):
                    navigate("problem_detail", problem_id=problem["id"])
    else:
        st.error(response["data"]["message"])

    st.subheader("创建新题目")
    with st.form("create_problem_form"):
        problem_id = st.text_input("题目ID")
        title = st.text_input("标题")
        description = st.text_area("题目描述")
        input_desc = st.text_area("输入描述")
        output_desc = st.text_area("输出描述")
        constraints = st.text_area("约束条件")
        time_limit = st.number_input("时间限制(秒)")
        memory_limit = st.number_input("内存限制(MB)")
        samples_json = st.text_area("样例列表")
        testcases_json = st.text_area("测试用例列表")

        samples = json.loads(samples_json) if samples_json else []
        testcases = json.loads(testcases_json) if testcases_json else []

        hint = st.text_area("提示信息")
        source = st.text_input("题目来源")
        tags = st.text_input("标签")
        author = st.text_input("作者")
        difficulty = st.text_input("难度")
        is_public = st.checkbox("是否公开题目", value=True)

        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        submitted = st.form_submit_button("创建题目")
        if submitted:
            problem_data = {
                "id": problem_id,
                "title": title,
                "description": description,
                "input_description": input_desc,
                "output_description": output_desc,
                "constraints": constraints,
                "time_limit": time_limit,
                "memory_limit": memory_limit,
                "samples": samples,
                "testcases": testcases,
                "hint": hint,
                "source": source,
                "tags": tag_list,
                "author": author,
                "difficulty": difficulty,
                "is_public": is_public,
            }

            response = api_request("POST", "/api/problems/", problem_data)
            if response["data"]["code"] == 200:
                st.success("题目创建成功")
                time.sleep(1)
                st.rerun()
            else:
                st.error(response["data"]["message"])

    if st.sidebar.button("退出登录"):
        api_request("POST", "/api/auth/logout")
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_id = ""
        st.session_state.role = ""
        navigate("login")


def problem_detail_page():
    "题目详情页面"
    problem_id = st.session_state.problem_id
    st.title(f"题目 {problem_id}")

    response = api_request("GET", f"/api/problems/{problem_id}")

    if response["data"]["code"] != 200:
        st.error(response["data"]["message"])
        if st.button("返回题目列表"):
            navigate("problems")
        return

    problem = response["data"]["data"]

    st.subheader(problem["title"])

    st.markdown("### 题目描述")
    st.write(problem["description"])

    st.markdown("### 输入描述")
    st.write(problem["input_description"])

    st.markdown("### 输出描述")
    st.write(problem["output_description"])

    st.markdown("### 约束条件")
    st.write(problem["constraints"])

    st.markdown("### 样例")
    for i, sample in enumerate(problem.get("samples", [])):
        st.markdown(f"**样例 {i+1} 输入**")
        st.code(sample["input"])
        st.markdown(f"**样例 {i+1} 输出**")
        st.code(sample["output"])

    if problem.get("testcases"):
        for i, testcase in enumerate(problem["testcases"]):
            st.markdown(f"**测试点 {i+1} 输入**")
            st.code(testcase["input"])
            st.markdown(f"**测试点 {i+1} 输出**")
            st.code(testcase["output"])

    st.markdown(f"### 限制")
    st.write(f"时间限制: {problem['time_limit']}秒")
    st.write(f"内存限制: {problem['memory_limit']}MB")

    st.markdown(f"### 限制")
    st.write(f"时间限制: {problem['time_limit']}秒")
    st.write(f"内存限制: {problem['memory_limit']}MB")

    st.subheader("提交代码")
    language = st.text_input("语言")
    code = st.text_area("代码")

    if st.button("提交"):
        response = api_request(
            "POST",
            "/api/submissions/",
            {"problem_id": problem_id, "language": language, "code": code},
        )
        if response["data"]["code"] == 200:
            submission_id = response["data"]["data"]["submission_id"]
            st.success(f"提交成功！提交ID: {submission_id}")
            st.info("正在评测，请等待...")
            navigate("submission_result", submission_id=submission_id)
        else:
            st.error(response["data"]["message"])


def submission_result_page(submission_id):
    "评测结果页面"
    st.title(f"提交结果 #{submission_id}")
    response = api_request("GET", f"/api/submissions/{submission_id}")
    if response["data"]["code"] != 200:
        st.error(response["data"]["message"])
        return

    data = response["data"]["data"]
    status = data["status"]

    if status == "pending":
        st.info(f"状态: {status}（请刷新页面或重新进入查看最新结果）")
    else:
        st.success(f"状态: {status} | 得分: {data['score']}/{data['counts']}")
        log_response = api_request("GET", f"/api/submissions/{submission_id}/log")
        if log_response["data"]["code"] == 200:
            log_data = log_response["data"]["data"]
            with st.expander("查看详细评测结果", expanded=True):
                for detail in log_data["details"]:
                    st.markdown(f"**测试点 #{detail['id']}**")
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"结果: {detail['result']}")
                    col2.write(f"时间: {detail.get('time', 'N/A')}s")
                    col3.write(f"内存: {detail.get('memory', 'N/A')}MB")
                    st.divider()
                st.write("**总评**")
                col1, col2 = st.columns(2)
                col1.write(f"总得分: {log_data['score']}/{log_data['counts']}")
        else:
            st.error(log_response["data"]["message"])
    col1, col2 = st.columns(2)
    if col1.button("返回题目"):
        navigate("problem_detail", st.session_state.problem_id)
    if col2.button("刷新结果"):
        st.rerun()


def main():
    st.sidebar.title(f"欢迎, {st.session_state.username or '访客'}")
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.sidebar.button("题目列表"):
            navigate("problems")
        if st.session_state.current_page == "problems":
            problems_page()
        elif st.session_state.current_page == "problem_detail":
            problem_detail_page()
        elif st.session_state.current_page == "submission_result":
            submission_result_page(st.session_state.submission_id)


main()
