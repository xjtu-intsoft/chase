cd data || exit

case "$(uname -s)" in
   Darwin)
     wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0" -O spider.zip
     ;;

   Linux)
     wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0" -O spider.zip
     ;;
esac

rm -rf /tmp/cookies.txt
unzip spider.zip
cp -r spider/database database
cd ..
python3 scripts/split_spider_by_db.py
rm data/spider.zip