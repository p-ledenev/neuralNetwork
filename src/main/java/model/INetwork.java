package model;

/**
 * Created by ledenev.p on 17.08.2015.
 */
public interface INetwork {

    void iteration();

    double getError();

    void setThreadCount(int count);

    void finishTraining();

    void initFrom(String fileName);
}
