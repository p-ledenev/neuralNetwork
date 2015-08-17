package model;

import org.apache.log4j.Logger;

/**
 * Created by DiKey on 17.08.2015.
 */
public class Log {

    public static Logger logger = Logger.getLogger(Log.class);

    public static void info(String message) {
        logger.info(message);
    }

    public static void debug(String message, Throwable e) {
        logger.debug(message, e);
    }
}
