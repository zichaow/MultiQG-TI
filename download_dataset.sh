if [ ! -d dataset ]; then
  mkdir dataset
fi
cd dataset

## Download pid_splits.csv
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OXlNBuW74dsrwYZIpQMshFqxkjcMPPgV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OXlNBuW74dsrwYZIpQMshFqxkjcMPPgV" -O pid_splits.json && rm -rf /tmp/cookies.txt

## Download problems.csv
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nJ86OLnF2C6eDoi5UOAdTAS5Duc0wuTl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nJ86OLnF2C6eDoi5UOAdTAS5Duc0wuTl" -O problems.json && rm -rf /tmp/cookies.txt

## Download test.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eyjFaHxbvEJZzdZILn3vnTihBNDmKcIj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eyjFaHxbvEJZzdZILn3vnTihBNDmKcIj" -O test.zip && rm -rf /tmp/cookies.txt
unzip test.zip

## Download train.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1swX4Eei1ZqrXRvM-JAZxN6QVwcBLPHV8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1swX4Eei1ZqrXRvM-JAZxN6QVwcBLPHV8" -O train.zip && rm -rf /tmp/cookies.txt
unzip train.zip

## Download val.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ijThWZc1tsoqGrOCWhYYj1HUJ48Hl8Zz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ijThWZc1tsoqGrOCWhYYj1HUJ48Hl8Zz" -O val.zip && rm -rf /tmp/cookies.txt
unzip val.zip

## Download version.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HIMhO4MqJBzsyn1jEuyQSzcUnYROsRRU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HIMhO4MqJBzsyn1jEuyQSzcUnYROsRRU" -O version.txt && rm -rf /tmp/cookies.txt

cd ..