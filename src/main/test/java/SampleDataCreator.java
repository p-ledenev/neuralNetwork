import model.Runner;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;

/**
 * Created by DiKey on 23.08.2015.
 */
public class SampleDataCreator {

    @Test
    public void create() throws Throwable {

        BufferedReader reader = new BufferedReader(new FileReader(Runner.dataPath + "usd_50inputs_app_2010.txt"));
        FileWriter writer = new FileWriter(Runner.dataPath + "usd_sample.txt");

        String line;
        for (int i = 0; (line = reader.readLine()) != null; i++) {
            if (i % 10000 == 0)

        }

    }
}
