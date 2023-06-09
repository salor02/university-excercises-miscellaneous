CREATE TABLE CLIENTI (
		EMAIL VARCHAR(20) NOT NULL,
		NOME VARCHAR(60) NOT NULL,
		COGNOME VARCHAR(60) NOT NULL,
		TELEFONO VARCHAR(20),
		PRIMARY KEY (EMAIL)
	);

CREATE TABLE FILM (
		CODICE CHAR(5) NOT NULL,
		TITOLO VARCHAR(60),
		REGISTA VARCHAR(60),
		VOTO INT CHECK (VOTO BETWEEN 1 AND 10),
		TRAMA VARCHAR(200),
		PRIMARY KEY (CODICE)
 	);

CREATE TABLE VISIONI (
		CODICE_FILM CHAR(5) NOT NULL REFERENCES FILM(CODICE),
		EMAIL VARCHAR(20) NOT NULL REFERENCES CLIENTI(EMAIL),
		DATA_VIS DATE,
		PRIMARY KEY (CODICE_FILM,EMAIL)
	);