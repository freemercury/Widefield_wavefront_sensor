for (i = 1; i < 74; i++) {
for (j = 0; j < 200; j++) {
open("D:/hyh/Project/LFM/data/realign_data_230406/"+i+"/realign/test_No"+j+".tif");
selectWindow("test_No"+j+".tif");
run("Remove Outliers...", "radius=1 threshold=75 which=Dark stack");
//setTool("rectangle");
makeRectangle(338, 81, 15, 11);
run("Remove Outliers...", "radius=1 threshold=125 which=Bright stack");
makeRectangle(277, 227, 11, 8);
run("Remove Outliers...", "radius=1 threshold=125 which=Bright stack");
makeRectangle(172, 90, 9, 6);
run("Remove Outliers...", "radius=1 threshold=125 which=Bright stack");
makeRectangle(417, 68, 4, 5);
run("Remove Outliers...", "radius=1 threshold=125 which=Bright stack");
run("Save");
close();
}}
