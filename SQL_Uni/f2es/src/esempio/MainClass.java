package esempio;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class MainClass {
	
	public static void main(String[] args){
		try{
			Class.forName("org.postgresql.Driver");  // Load the Driver
			Connection conn = DriverManager.getConnection( "jdbc:postgresql://localhost:5432/Studenti", "postgres", "supersal02" );
			Statement stmt = conn.createStatement();
            
            String sql;
            ResultSet rs;
            int numRows;

            //es 1a
            sql = "CREATE TABLE IF NOT EXISTS studenti(matr INTEGER PRIMARY KEY, cognome VARCHAR(20), nome VARCHAR(20))";
            numRows = stmt.executeUpdate(sql);

            /*sql = "INSERT INTO studenti VALUES(1,'rossi','mario'),(2,'bianchi','sergio')";
			numRows = stmt.executeUpdate(sql);*/

            //es 1b
            sql = "SELECT * FROM studenti";
            rs = stmt.executeQuery(sql);

            System.out.println("[RESULT] of " + sql);
            while(rs.next()){
                System.out.println(rs.getInt("matr") + " " + rs.getString("cognome") + " " + rs.getString("nome"));
            }
            rs.close();

            //es 2a
            sql = "SELECT * FROM studenti WHERE nome = ? and matr > ?";
            PreparedStatement pstmt = conn.prepareStatement(sql);

            pstmt.setString(1, "sergio");
            pstmt.setInt(2, 0);
            rs = pstmt.executeQuery();

            System.out.println("[RESULT] of " + pstmt.toString());
            while(rs.next()){
                System.out.println(rs.getInt("matr") + " " + rs.getString("cognome") + " " + rs.getString("nome"));
            }
            rs.close();
			
			conn.close();
		}
		catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		catch (SQLException e) {
			e.printStackTrace();
		}
		
	}
}
