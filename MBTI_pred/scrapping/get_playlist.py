import sys
import json
import webbrowser

# 명령행 인수를 통해 MBTI 유형을 전달받습니다
if len(sys.argv) < 2:
    print("MBTI 유형을 전달해야 합니다.")
    sys.exit(1)

mbti_type = sys.argv[1]

# mbti_playlists.json 파일을 읽어옵니다
with open('/Users/ljeonghyeon/Desktop/객체지향_6조/MBTI_pred/scrapping/mbti_playlists.json', 'r') as f:
    mbti_playlists = json.load(f)

# 예측된 MBTI 유형에 해당하는 플레이리스트 URL을 가져옵니다
if mbti_type in mbti_playlists:
    playlists = mbti_playlists[mbti_type]["playlists"]["items"]
    if playlists:
        # 첫 번째 플레이리스트 URL을 자동으로 실행
        first_playlist_url = playlists[0]['external_urls']['spotify']
        webbrowser.open(first_playlist_url)
        
        # 나머지 플레이리스트 URL 출력
        for playlist in playlists:
            print(f"플레이리스트 이름: {playlist['name']}")
            print(f"플레이리스트 링크: {playlist['external_urls']['spotify']}")
            print("\n")
    else:
        print(f"{mbti_type} 유형에 해당하는 플레이리스트가 없습니다.")
else:
    print(f"{mbti_type} 유형에 해당하는 플레이리스트를 찾을 수 없습니다.")
