# AlyAc
#### 사용자가 스마트폰으로 알약의 사진을 찍어서 전송하면, 해당 알약의 이름, 주의사항 등을 알려주는 카카오톡 챗봇

________________

### 통신 방법

![image](https://user-images.githubusercontent.com/67954861/166141199-b366a28c-3530-4215-80ce-ad664c72f61a.png)
1. 사용자가 카카오톡 챗봇으로 알약 사진을 보낸다.
2. 카카오톡 서버가 해당 사진을 받아서, 정보를 json파일로 변환해서 컴퓨터비전연구실 서버로 보낸다.
3. 컴퓨터비전연구실 서버에는 이미 train 데이터셋을 이용해서 학습한 모델이 있다. 해당 모델을 이용해서 받은 정보에 있는 영상으로 분류를 진행한다.
4. 분류 후, 결과를 다시 json파일로 변환해서 카카오톡 서버로 보낸다.
5. 카카오톡 서버에서 사용자에게 분류 결과를 알려준다. 해당 분류 결과에 알맞은 정보를 제공한다.

### 작품 시연
https://youtu.be/obC3F3zMJpY


___________

## 1. 세부 구현 

### 예외 처리
카카오에서 메시지마다 5초의 답장 제한시간을 걸어둠.  
5초 안에 결과를 사용자에게 전송하지 않으면, 사용자는 결과를 받지 못함(5초가 지나면 답장을 안함)   
&rarr; 어떠한 문제로 (주로 네트워크 문제) 5초 안에 결과를 못받는다면 time out 메시지 출력(사용자가 계속 기다리게 되는 문제 해결)  

![image](https://user-images.githubusercontent.com/67954861/166141722-5484aeb3-f5f1-4946-ad2c-ccc9f526cd14.png)

__________

### 선택지 추가
정확도에 따라 1~3개의 선택지 제공 &rarr; 사람이 판단을 할 수 있도록 설계  

![image](https://user-images.githubusercontent.com/67954861/166141900-08bad352-eec8-47b0-9492-8a51e7a52538.png)

________

### 최악의 경우 대비
사용자의 알약을 예축 못하는 경우도 존재 가능 &rarr; 직접 찾기 누르면 약학정보원의 알약 검색 페이지로 연결   

![image](https://user-images.githubusercontent.com/67954861/166141885-98f0fd84-2c06-497c-a3cb-4fe6c474dca0.png)

_________

## 2. 문제점과 해결방안 

#### Problem1. 다수의 사용자  

![image](https://user-images.githubusercontent.com/67954861/166142122-5b0bb7c8-8538-4fca-818d-5b48be199e85.png) 

다수의 사용자가 동시에 사용 시 충돌. 
원인 : 폴더를 공유해서 사용하기 때문

#### Solution1. Key 할당 

![image](https://user-images.githubusercontent.com/67954861/166142156-f2824ba8-d79d-4af8-95fc-8bb3a131400c.png) 

각각 Key 할당. 자신의 Key에 해당하는 폴더 사용

_______

#### Problem2. 데이터 확보

![image](https://user-images.githubusercontent.com/67954861/166142212-250f3f34-bd27-4eb6-a5c0-11d0ee03a5f4.png) 

실제 사진과 인터넷 이미지의 차이

실제 사진은 배경과 그림자, 빛, 각도 등이 다양한 반면, 인터넷 이미지들은 배경이 다양하지 않고, 그림자도 없고, 데이터가 너무 적다는 문제


#### Solution2. 데이터 구축

![image](https://user-images.githubusercontent.com/67954861/166142255-b9ed81c6-df9b-4308-ab6e-0c4a2bf7cb51.png)

직접 데이터를 만들어서 사용. 빛의 방향, 그림자, 배경 등이 다른 다양한 이미지 확보  
그러나, 데이터를 만들지 않은 알약은 예측할 수 없다는 문제점..


## 3. 발전 방향 
#### 1. 데이터 확보   

외부 기관(전북대 병원)과 연계로 추가 데이터 확보  
사용자가 전송한 입력 이미지를 사용해서 추가 데이터 확보   

#### 2. 물체 분할(Segmentation)   
여러개의 알약을 한번에 인식하도록 Segmentation으로 알약 인식   

![image](https://user-images.githubusercontent.com/67954861/166142533-b9d541aa-c5d9-4b12-a147-a2862b1848ce.png)











