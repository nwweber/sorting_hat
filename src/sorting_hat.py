from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypeAlias, Union

import pandas
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, BoundedLinearExpression
from pandas import DataFrame
import click

StudentPreferences: TypeAlias = Dict[str, List[str]]

EXAMPLE_STUDENT_PREFERENCES_FILENAME: str = "example_student_preferences.csv"
EXAMPLE_COURSE_CAPACITY_FILENAME: str = "example_course_capacity.csv"
EXAMPLE_SOLUTION_FILENAME: str = "example_assignment_solution.csv"
EXAMPLE_FILE_ENCODING: str = "utf-8"


class Courses:
    valid_fields: List[str] = ["name", "min_size", "max_size"]

    @classmethod
    def make_from_file(cls, file_path: Path, encoding: Union[str, None]) -> Courses:
        course_info: DataFrame = pandas.read_csv(file_path, encoding=encoding)
        return Courses(course_info)

    def __init__(self, course_info: DataFrame):
        if not set(self.valid_fields) == set(course_info.columns):
            raise ValueError(
                f"expected fields {self.valid_fields} but received {course_info.columns}"
            )
        self.course_info: DataFrame = course_info

    def __len__(self) -> int:
        return len(self.course_info)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        other: Courses
        return self.course_info.equals(other.course_info)

    def __str__(self):
        return str(self.course_info)

    def get_all_course_names(self) -> List[str]:
        return self.course_info["name"].to_list()

    def get_max_students_by_course_name(self, course_name: str) -> int:
        field: str = 'max_size'
        value = self.query_single_record_field_by_course_name(course_name, field)
        return value

    def get_min_students_by_course_name(self, course_name: str):
        field: str = 'min_size'
        value = self.query_single_record_field_by_course_name(course_name, field)
        return value

    def query_single_record_field_by_course_name(self, course_name: str, field: str):
        records: DataFrame = self.course_info.query(f"name == '{course_name}'")
        assert (
            len(records) == 1
        ), f"found {len(records)} entries for course {course_name}, expected exactly 1"
        value: int = records[field].squeeze()
        return value


def get_example_problem():
    student_preferences: StudentPreferences = read_student_preferences_file(
        Path(EXAMPLE_STUDENT_PREFERENCES_FILENAME), EXAMPLE_FILE_ENCODING
    )
    course_max_students: Courses = Courses.make_from_file(
        Path(EXAMPLE_COURSE_CAPACITY_FILENAME), EXAMPLE_FILE_ENCODING
    )
    return student_preferences, course_max_students


class CourseAssignmentVariables:
    def __init__(self, initial_variables: List[Tuple[str, str, cp_model.IntVar]]):
        self.variables: DataFrame = DataFrame(
            data=initial_variables, columns=["student", "course", "variable"]
        )

    def __eq__(self, other: Any):
        if not isinstance(other, self.__class__):
            return False
        # not sure how to meaningfully compare the IntVar objects, so dropping those for now
        return self.variables[["student", "course"]].equals(
            other.variables[["student", "course"]]
        )

    def get_by_student_name(self, name: str) -> List[cp_model.IntVar]:
        student_vars: List[cp_model.IntVar] = self.variables.query(
            f"student == '{name}'"
        )["variable"].to_list()
        return student_vars

    def get_by_course_name(self, name: str) -> List[IntVar]:
        course_vars: List[cp_model.IntVar] = self.variables.query(
            f"course == '{name}'"
        )["variable"].to_list()
        return course_vars

    def get_by_student_name_and_courses(
        self, student_name: str, course_names: List[str]
    ):
        variables: List[IntVar] = self.variables[
            (self.variables["student"] == student_name)
            & (self.variables["course"].isin(course_names))
        ]["variable"].to_list()
        if len(variables) == 0:
            raise ValueError(
                f"no variables for student {student_name}, courses {course_names}"
            )
        return variables

    def get_all(self) -> List[IntVar]:
        return self.variables["variable"].to_list()

    def report_final_assignments(self, solver: cp_model.CpSolver) -> DataFrame:
        solver_decisions: List[bool] = [
            solver.Value(var) == 1 for var in self.variables["variable"]
        ]
        return self.variables[solver_decisions][["student", "course"]].reset_index(
            drop=True
        )


def generate_course_assignment_variables(
    students: StudentPreferences, courses: Courses, model: CpModel
) -> CourseAssignmentVariables:
    student_names: List[str] = list(students.keys())
    course_names: List[str] = courses.get_all_course_names()
    initial_variables: List[Tuple[str, str, cp_model.IntVar]] = [
        (
            student_name,
            course_name,
            model.NewBoolVar(f"{student_name} in {course_name}"),
        )
        for student_name in student_names
        for course_name in course_names
    ]
    assignments = CourseAssignmentVariables(initial_variables)
    return assignments


def solve_example_problem() -> None:
    # see for example: https://developers.google.com/optimization/cp/cp_example
    solve_from_and_to_files(
        Path(EXAMPLE_COURSE_CAPACITY_FILENAME),
        Path(EXAMPLE_STUDENT_PREFERENCES_FILENAME),
        Path(EXAMPLE_SOLUTION_FILENAME),
        EXAMPLE_FILE_ENCODING,
    )


def generate_constraints_only_preferred_courses(
    assignment_variables: CourseAssignmentVariables,
    course_max_students: Courses,
    student_preferences: StudentPreferences,
) -> List[BoundedLinearExpression]:
    only_preferred_courses_constraints: List[BoundedLinearExpression] = []
    all_course_name_set: set = set(course_max_students.get_all_course_names())
    student_names: List[str] = list(student_preferences.keys())
    for student_name in student_names:
        student_preferred_course_set: set = set(student_preferences[student_name])
        non_preferred_courses: set = all_course_name_set - student_preferred_course_set
        non_preferred_assign_vars: List[
            IntVar
        ] = assignment_variables.get_by_student_name_and_courses(
            student_name, list(non_preferred_courses)
        )
        for av in non_preferred_assign_vars:
            only_preferred_courses_constraints.append(av == 0)
    return only_preferred_courses_constraints


def generate_constraints_max_students_per_course(
    assignment_variables: CourseAssignmentVariables, courses: Courses,
) -> List[BoundedLinearExpression]:
    course_names: list[str] = courses.get_all_course_names()
    max_students_per_course_constraints: List[BoundedLinearExpression] = []
    for course_name in course_names:
        course_max_nr_students: int = courses.get_max_students_by_course_name(
            course_name
        )
        variables_for_course: List[IntVar] = assignment_variables.get_by_course_name(
            course_name
        )
        constraint: BoundedLinearExpression = sum(
            variables_for_course
        ) <= course_max_nr_students
        max_students_per_course_constraints.append(constraint)
    return max_students_per_course_constraints


def generate_constraints_exactly_one_course_per_student(
    assignment_variables: CourseAssignmentVariables,
    student_preferences: StudentPreferences,
) -> List[BoundedLinearExpression]:
    student_names: list = list(student_preferences.keys())
    exactly_one_course_constraints: List[BoundedLinearExpression] = []
    for student_name in student_names:
        variables_for_student: List[
            cp_model.IntVar
        ] = assignment_variables.get_by_student_name(student_name)
        constraint: BoundedLinearExpression = sum(variables_for_student) == 1
        exactly_one_course_constraints.append(constraint)
    return exactly_one_course_constraints


def generate_cost(
    students: StudentPreferences, course_assignments: CourseAssignmentVariables
) -> BoundedLinearExpression:
    cost_terms: List[BoundedLinearExpression] = []
    for student_name, course_list in students.items():
        # note that courses in list are ordered by preference
        for preference_index, course in enumerate(course_list):
            assign_var: IntVar = course_assignments.get_by_student_name_and_courses(
                student_name, [course]
            )[0]
            cost_terms.append(preference_index * assign_var)
    cost: BoundedLinearExpression = sum(cost_terms)
    return cost


def read_student_preferences_file(
    file_path: Path, encoding: Union[str, None]
) -> StudentPreferences:
    out: StudentPreferences = {}
    with file_path.open("r", encoding=encoding) as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in reader:
            student, courses = row[0], row[1:]
            out[student] = courses
    return out


def solve(students: StudentPreferences, courses: Courses) -> Union[DataFrame, None]:
    model = cp_model.CpModel()
    assignment_variables: CourseAssignmentVariables = generate_course_assignment_variables(
        students, courses, model
    )

    exactly_one_course_constraints = generate_constraints_exactly_one_course_per_student(
        assignment_variables, students
    )
    max_students_per_course_constraints = generate_constraints_max_students_per_course(
        assignment_variables, courses
    )
    only_preferred_courses_constraints = generate_constraints_only_preferred_courses(
        assignment_variables, courses, students
    )

    all_constraints: List[
        BoundedLinearExpression
    ] = exactly_one_course_constraints + max_students_per_course_constraints + only_preferred_courses_constraints
    for constraint in all_constraints:
        model.Add(constraint)

    # doesn't fit above schema of separating constraint creation and the actual addition to the model
    # reason: min nr students involves logical or operation, which is tricky to implement in ortools and
    # needs access to the actual model
    add_constraints_to_model_min_nr_students(assignment_variables, courses, model)

    total_cost = generate_cost(students, assignment_variables)
    model.Minimize(total_cost)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Minimum of objective function: {solver.ObjectiveValue()}\n")
        all_assignment_variables: List[IntVar] = assignment_variables.get_all()
        for assignment in all_assignment_variables:
            print(f"{assignment} = {solver.Value(assignment)}")
        final_assignment_report: DataFrame = assignment_variables.report_final_assignments(
            solver
        )
        print("Found this assignment of students to courses:")
        print(final_assignment_report)
        return final_assignment_report
    else:
        print("No solution found.")
        return None


def solve_from_and_to_files(
    capacity_path: Path,
    student_path: Path,
    solution_path: Path,
    encoding: Union[str, None],
) -> None:
    students: StudentPreferences = read_student_preferences_file(student_path, encoding)
    courses: Courses = Courses.make_from_file(capacity_path, encoding)
    solution: Union[None, DataFrame] = solve(students, courses)
    if solution is not None:
        solution.to_csv(solution_path, index=False, encoding=encoding)
        print(f"Saved solution to {solution_path}")
    return None


@click.command()
@click.argument("capacity_file")
@click.argument("student_file")
@click.argument("solution_file")
@click.option(
    "--encoding",
    help="check here for possible values: https://stackoverflow.com/a/25584253",
    default=None,
)
def solve_from_command_line_args(
    capacity_file: str, student_file: str, solution_file: str, encoding: str
) -> None:
    """
    Read course capacities from CAPACITY_FILE, read student preferences from STUDENT_FILE,
    attempt to solve optimally and write output to SOLUTION_FILE (will be created if it does not exist).

    Files are read/writen using character ENCODING if given. If omitted, whatever Python uses by default on your system
    will be used. If you see some hickups due to 'cannot decode character' etc this might be  place to start looking.
    Have a look here for possible values for this option: https://stackoverflow.com/a/25584253

    Input files should be CSV files. Output will be written as CSV as well.
    """
    cap_file: Path = Path(capacity_file)
    stud_file: Path = Path(student_file)
    sol_file: Path = Path(solution_file)
    solve_from_and_to_files(cap_file, stud_file, sol_file, encoding)


if __name__ == "__main__":
    solve_from_command_line_args()


def add_constraints_to_model_min_nr_students(
    assignment_variables: CourseAssignmentVariables, courses: Courses, model: CpModel
) -> None:
    course_names: list[str] = courses.get_all_course_names()
    for course_name in course_names:
        course_min_nr_students: int = courses.get_min_students_by_course_name(
            course_name
        )
        variables_for_course: List[IntVar] = assignment_variables.get_by_course_name(
            course_name
        )
        n_students_assigned_to_course = sum(variables_for_course)
        either_or: IntVar = model.NewBoolVar(f'either 0 or min_nr_students for {course_name}')
        enough_students: BoundedLinearExpression = (n_students_assigned_to_course >= course_min_nr_students)
        no_students: BoundedLinearExpression = (n_students_assigned_to_course == 0)
        # below is a way to express that EITHER course should have 0 students OR
        # at least min_nr_students. unhandy to express in ortools, see here for discussion:
        # https://or.stackexchange.com/questions/4332/how-to-add-logical-or-constraint-in-or-tools
        model.Add(no_students).OnlyEnforceIf(either_or)
        model.Add(enough_students).OnlyEnforceIf(either_or)
