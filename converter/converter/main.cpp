#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>

using namespace std;
int main()
{
    string path = "/Users/sorush/My Drive/Documents/Educational/TAMU/Research/Trial/Data/11-5-21-11-15-21/hk+cm/p";
    for (int counter = 1; counter < 5; counter++)
    {
        string lineContent, allContent;
        fstream input_file;
        input_file.open((path +to_string(counter)+ "_cm_all.csv"), ios::in);
        char delimiter = ',';
        getline(input_file, lineContent);
        allContent += "Time,Ax,Ay,Az,Rx,Ry,Rz,Yaw,Roll,Pitch\n";
        while(getline(input_file, lineContent))
        {
            string tempStr;
            size_t pos1, pos2;
            pos1 = 0;
            pos2 = 0;
            for (int i = 0; i < 2; i++)
            {
                pos1 = lineContent.find(delimiter, pos1 + 1);
                pos2 = lineContent.find(delimiter, pos1 + 1);
            }
            allContent += lineContent.substr(pos1 + 1, 26) + lineContent.substr(pos2) + "\n";
        }

        input_file.close();
        fstream out_file;
        out_file.open((path+to_string(counter)+ "_cm_all_modified.csv"), ios::out);
        out_file << allContent;
        out_file.close();
    }
    return 0;
}
