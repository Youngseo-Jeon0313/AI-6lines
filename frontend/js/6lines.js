window.onload = function () {
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');
  const margin = 30;
  const cw = (ch = canvas.width = canvas.height = 600 + margin * 2);
  const row = 18; // 바둑판 선 개수
  const rowSize = 600 / row; // 바둑판 한 칸의 너비
  const dolSize = 13;  // 바둑돌 크기
  let count = 0;
  let msg = document.querySelector('.message');
  let btn1 = document.querySelector('#reload');
  let btn2 = document.querySelector('#withdraw');
  let board = new Array(Math.pow(row + 1, 2)).fill(-1); // 144개의 배열을 생성해서 -1로 채움
  let history = new Array();
  let checkDirection = [
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
  ];
  const blackWinScreen = document.querySelector('.winShow1');
  const whiteWinScreen = document.querySelector('.winShow2');
  let audio1 = new Audio('tik.mp3');
  let audio2 = new Audio('beep.wav');
  let audio3 = new Audio('ending.wav');
  let audio4 = new Audio('plz.m4a');
  let audio5 = new Audio('oneMore.m4a');

  // '한판 더' 버튼 누르면 페이지 reload
  btn1.addEventListener('mouseup', () => {
    setTimeout(() => {
      location.reload();
    }, 2000);
  });
  // 무르기 버튼 누르면 withdraw() 함수 실행
  btn2.addEventListener('mouseup', () => {
    audio4.play();
    withdraw();
  });

  draw(); // 시작하면서 빈 바둑판 그리기

  // 배열을 콘솔창에 grid로 보여주는 함수.
  // 코딩하면서 바둑판이 어떻게 그려지는지 콘솔창에서 확인하려는 목적이고, 게임과는 관계 없음.
  function indexView(m) {
    let s = '\n';
    let c = 0;
    for (let e of m) {
      s += `${e} `;
      if (c % (row + 1) === row) s += '\n'; //줄바꿈 문자 삽입 
      c++;
    }
    return s;
  }

  // x,y 좌표를 배열의 index값으로 변환
  let xyToIndex = (x, y) => {
    return x + y * (row + 1);
  };

  // 배열 index값을 x,y좌표로 변환
  let indexToXy = (i) => {
    w = Math.sqrt(board.length);
    x = i % w;
    y = Math.floor(i / w);
    return [x, y];
  };

  // 바둑판 그리기 함수
  function draw() {
    ctx.fillStyle = '#e38d00';
    ctx.fillRect(0, 0, cw, ch);
    for (let x = 0; x < row; x++) {
      for (let y = 0; y < row; y++) {
        let w = (cw - margin * 2) / row;
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.strokeRect(w * x + margin, w * y + margin, w, w);
      }
    }

    // 화점에 점 찍기
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        ctx.fillStyle = 'black';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(
          (3 + a) * rowSize + margin + a * 5 * rowSize,
          (3 + b) * rowSize + margin + b * 5 * rowSize,
          dolSize / 3,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
    }
  }

  // 방금 둔 바둑돌에 사각 표시
  drawRect = (x, y) => {
    let w = rowSize/2;
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 3;
    ctx.strokeRect(
      x * rowSize + margin - w,
      y * rowSize + margin - w,
      w + rowSize/2,
      w + rowSize/2
    );
  };

  //바둑알 그리기. 실제로는 바둑판까지 매번 통째로 그려줌
  drawCircle = (x, y) => {
    draw();
    drawRect(x, y);
    for (i = 0; i < board.length; i++) {
      // 모든 눈금의 돌의 유무,색깔 알아내기
      let a = indexToXy(i)[0];
      let b = indexToXy(i)[1];

      if (board[xyToIndex(a, b)] == 1) {
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(
          a * rowSize + margin,
          b * rowSize + margin,
          dolSize,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
      if (board[xyToIndex(a, b)] == 2) {
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(
          a * rowSize + margin,
          b * rowSize + margin,
          dolSize,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
    }

    checkWin(x, y); // 돌이 5개 연속 놓였는지 확인 함수 실행

    let boardCopy = Object.assign([], board);
    history.push(boardCopy); //무르기를 위해서 판 전체 모양을 배열에 입력
  };

  // 물르기 함수
  withdraw = () => {
    history.pop(); // 무르면서 가장 최근 바둑판 모양은 날려버림
    lastBoard = history.slice(-1)[0]; // 바둑판 마지막 모양
    board = lastBoard;
    count--; // 흑,백 차례를 한 수 뒤로 물림

    draw();

    // 직전 판의 모양으로 바둑판 다시 그리기
    for (i = 0; i < lastBoard.length; i++) {
      let a = indexToXy(i)[0];
      let b = indexToXy(i)[1];

      if (lastBoard[xyToIndex(a, b)] == 1) {
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(
          a * rowSize + margin,
          b * rowSize + margin,
          dolSize,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
      if (lastBoard[xyToIndex(a, b)] == 2) {
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(
          a * rowSize + margin,
          b * rowSize + margin,
          dolSize,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
    }
  };

  // 승패 판정 함수
  function checkWin(x, y) {
    let thisColor = board[xyToIndex(x, y)]; // 마지막 둔 돌의 색깔이 1(흑),2(백)인지...
    //가로,세로와 대각선 두 방향, 총 네 방향 체크
    for (k = 0; k < 4; k++) {
      winBlack = 1;   winWhite = 1;
      // 놓여진 돌의 양쪽 방향으로
      for (j = 0; j < 2; j++) {
        // 5개씩의 돌의 색깔을 확인
        for (i = 1; i < 5; i++) {
          let a = x + checkDirection[k + 4 * j][0] * i;
          let b = y + checkDirection[k + 4 * j][1] * i;
          if (board[xyToIndex(a, b)] == thisColor) {
            // 색깔에 따라서 흑,백의 숫자를 하나씩 증가
            switch (thisColor) {
              case 1: winBlack++; break;
              case 2: winWhite++; break;
            }
          } else { break; }
        }
      }
      // 연속 돌이 5개이면 승리
      if (winBlack == 5) {winShow(1);}
      if (winWhite == 5) {winShow(2);}
    }
  } // 승리확인 함수 끝

  // 승리 화면 표시
  function winShow(x) {
    switch (x) {
      case 1:
        // 음악이 재생되도록 약간의 시차를 두고 화면 표시
        setTimeout(() => {
          blackWinScreen.style.visibility = 'visible';
          blackWinScreen.style.zIndex = 2;
          const troImg = document.querySelector('#trophyImg');
          troImg.style.animationName = 'trophy';
        }, 300);
        break;
      case 2:
        // 음악이 재생되도록 약간의 시차를 두고 화면 표시
        setTimeout(() => {
          whiteWinScreen.style.visibility = 'visible';
          whiteWinScreen.style.zIndex = 2;
          const troImg = document.querySelector('#trophyImg2');
          troImg.style.animationName = 'trophy';
        }, 300);
        break;
    }
  }

  // 마우스 클릭한 위치를 정확한 눈금 위치로 보정
  document.addEventListener('mouseup', (e) => {
    if (e.target.id == 'canvas') {
      let x = Math.round(Math.abs(e.offsetX - margin) / rowSize);
      let y = Math.round(Math.abs(e.offsetY - margin) / rowSize);
      console.log(e.offsetX, e.offsetY, x, y);
      if (
        e.offsetX > 10 &&
        e.offsetX < 640 &&
        e.offsetY > 10 &&
        e.offsetY < 640
      ) {
        // 이미 돌이 놓여진 자리이면 비프음 출력
        if (board[xyToIndex(x, y)] != -1) {
          audio2.play();
        } else {
          // 비어있는 자리이면, 순서에 따라서 흑,백 구분해서 그리는 함수 실행
          count % 2 == 0
            ? (board[xyToIndex(x, y)] = 1)
            : (board[xyToIndex(x, y)] = 2);
          count++;
          drawCircle(x, y);
        }
      }
    }
  });
};;