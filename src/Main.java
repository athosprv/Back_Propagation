/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */



import java.io.*;


public class Main {
        public static void main(String[] args) throws Exception
    {
        BufferedReader instances = new BufferedReader(new FileReader("activation.txt"));
        BufferedReader weights = new BufferedReader(new FileReader("parameters.txt"));

       Solution.run(instances,weights);

        instances.close();
        weights.close();

    }

}
