import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypeAlias, Union
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, BoundedLinearExpression
from pandas import DataFrame
import click

StudentPreferences: TypeAlias = Dict[str, List[str]]
CourseCapacity: TypeAlias = Dict[str, int]

EXAMPLE_STUDENT_PREFERENCES_FILENAME: str = "example_student_preferences.csv"
EXAMPLE_COURSE_CAPACITY_FILENAME: str = "example_course_capacity.csv"
EXAMPLE_SOLUTION_FILENAME: str = "example_assignment_solution.csv"


def get_example_problem():
    student_preferences: StudentPreferences = read_student_preferences_file(
        Path(EXAMPLE_STUDENT_PREFERENCES_FILENAME)
    )
    course_max_students: CourseCapacity = read_course_capacity_file(
        Path(EXAMPLE_COURSE_CAPACITY_FILENAME)
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
    students: StudentPreferences, courses: CourseCapacity, model: CpModel
) -> CourseAssignmentVariables:
    student_names: List[str] = list(students.keys())
    course_names: List[str] = list(courses.keys())
    initial_variables: List[Tuple[str, str, cp_model.IntVar]] = [
        (
            student_name,
            course_name,
            model.NewIntVar(0, 1, f"{student_name} in {course_name}"),
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
    )


def generate_constraints_only_preferred_courses(
    assignment_variables: CourseAssignmentVariables,
    course_max_students: CourseCapacity,
    student_preferences: StudentPreferences,
) -> List[BoundedLinearExpression]:
    only_preferred_courses_constraints: List[BoundedLinearExpression] = []
    all_course_name_set: set = set(course_max_students.keys())
    student_names: List[str] = list(student_preferences.keys())
    for student_name in student_names:
        student_preferred_course_set: set = set(student_preferences[student_name])
        non_preferred_courses: set = all_course_name_set - student_preferred_course_set
        non_preferred_assign_vars: List[
            IntVar
        ] = assignment_variables.get_by_student_name_and_courses(
            student_name, non_preferred_courses
        )
        for av in non_preferred_assign_vars:
            only_preferred_courses_constraints.append(av == 0)
    return only_preferred_courses_constraints


def generate_constraints_max_students_per_course(
    assignment_variables: CourseAssignmentVariables,
    course_max_students: CourseCapacity,
) -> List[BoundedLinearExpression]:
    course_names: list[str] = list(course_max_students.keys())
    max_students_per_course_constraints: List[BoundedLinearExpression] = []
    for course_name in course_names:
        course_max_nr_students: int = course_max_students[course_name]
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


def read_student_preferences_file(pref_file_path: Path) -> StudentPreferences:
    preferences: StudentPreferences = {}
    with pref_file_path.open("r") as f:
        pref_reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in pref_reader:
            student, courses = row[0], row[1:]
            preferences[student] = courses
    return preferences


def read_course_capacity_file(capacity_file_path: Path) -> CourseCapacity:
    capacities: CourseCapacity = {}
    with capacity_file_path.open("r") as f:
        capacity_reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in capacity_reader:
            course, capacity = row[0], int(row[1])
            capacities[course] = capacity
    return capacities


def solve(
    students: StudentPreferences, courses: CourseCapacity
) -> Union[DataFrame, None]:
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
    capacity_path: Path, student_path: Path, solution_path: Path
) -> None:
    students: StudentPreferences = read_student_preferences_file(student_path)
    courses: CourseCapacity = read_course_capacity_file(capacity_path)
    solution: Union[None, DataFrame] = solve(students, courses)
    if solution is not None:
        solution.to_csv(solution_path, index=False)
        print(f"Saved solution to {solution_path}")
    return None


@click.command()
@click.argument('capacity_file')
@click.argument('student_file')
@click.argument('solution_file')
def solve_from_command_line_args(capacity_file: str, student_file: str, solution_file: str) -> None:
    """
    Read course capacities from CAPACITY_FILE, read student preferences from STUDENT FILE,
    attempt to solve optimally and write output to SOLUTION_FILE (will be created if it does not exist).

    Input files should be CSV files. Output will be written as CSV as well.
    """
    cap_file: Path = Path(capacity_file)
    stud_file: Path = Path(student_file)
    sol_file: Path = Path(solution_file)
    solve_from_and_to_files(cap_file, stud_file, sol_file)


if __name__ == '__main__':
    solve_from_command_line_args()
