# AI 서버 - Uvicorn 로컬 서버 실행 진입점
import uvicorn


def main():
    uvicorn.run("ai_server.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
